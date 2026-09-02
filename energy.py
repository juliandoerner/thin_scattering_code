"""Energy-norm, mesh-size and critical-resolution helpers."""

import numpy as np
from mpi4py import MPI
import dolfinx as dfx
import ufl

from mesh import CELL_ID


def energy_norm(w, k, mesh, dx, ds_scat, comm):
    """Return the energy norm of w over the air region plus obstacle boundary."""
    n = ufl.FacetNormal(mesh)
    P = ufl.Identity(mesh.geometry.dim) - ufl.outer(n, n)
    gt = ufl.dot(P, ufl.grad(w))
    form = dfx.fem.form(
        k**2 * ufl.inner(w, w) * dx
        + ufl.inner(ufl.grad(w), ufl.grad(w)) * dx
        + (1.0 / k) * ufl.inner(gt, gt) * ds_scat
        + k * ufl.inner(w, w) * ds_scat
    )
    val = comm.allreduce(dfx.fem.assemble_scalar(form), MPI.SUM)
    return float(np.sqrt(np.real(val)))


def h_max(mesh, cell_tags, comm):
    """Return the largest cell diameter over the physical cells."""
    tdim = mesh.topology.dim
    phys = cell_tags.find(CELL_ID.PHYS)
    hs = mesh.h(tdim, phys)
    return float(comm.allreduce(hs.max() if hs.size else 0.0, MPI.MAX))


def mie_callable(mie, k):
    """Wrap the Mie scattered field as an interpolation callable."""
    def f(x):
        pts = np.column_stack([x[0], x[1]])
        return mie.u_scattered(float(k), pts).astype(np.complex128)
    return f


def mie_grad_callable(mie, k):
    """Wrap the Mie scattered-field gradient as an interpolation callable."""
    def f(x):
        pts = np.column_stack([x[0], x[1]])
        g = mie.grad_scattered(float(k), pts)
        return g.T.astype(np.complex128)
    return f


def find_crossing(kh, ratio, thresh):
    """Find the coarsest kh where the FEM/best-approx ratio first exceeds thresh.

    Args:
        thresh: "close enough" tolerance; kh_crit is the coarsest kh with
            ratio still at or below this value.
    """
    idx = np.argsort(kh)
    kh = np.asarray(kh)[idx]
    ratio = np.asarray(ratio)[idx]
    above = ratio > thresh
    if not above.any():
        return kh[-1], "resolved_all"
    if above.all():
        return kh[0], "polluted_all"
    khc = None
    for i in range(len(kh) - 1):
        if ratio[i] <= thresh < ratio[i + 1]:
            t = (thresh - ratio[i]) / (ratio[i + 1] - ratio[i])
            khc = float(np.exp(np.log(kh[i]) + t * (np.log(kh[i + 1]) - np.log(kh[i]))))
    if khc is None:
        khc = float(kh[np.where(~above)[0][-1]])
    return khc, "ok"
