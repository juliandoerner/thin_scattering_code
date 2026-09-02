#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manufactured-solution validation for the Robin-truncated scattering solver.

The exact Mie scattered field is, by construction, the exact solution of the
discrete problem, so the FEM energy error should converge at rate O(h^p).

Run (complex PETSc required)::

    source dolfinx-complex-mode
    python test_robin.py            # serial
    mpirun -n 2 python test_robin.py
"""

import sys

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx as dfx
import ufl

from mesh import build_robin_mesh, FACET_ID
from incident import IncidentPlaneWave
from solver import RobinScatteringSolver
from mie import MieWentzellCylinder
from energy import energy_norm, h_max, mie_callable, mie_grad_callable

assert np.dtype(PETSc.ScalarType).kind == "c", "activate: source dolfinx-complex-mode"


def energy_error(p, k, lc, beta, alpha, r_s, L, order, inc, mie, comm):
    """FEM energy-norm error against the exact Mie field for one (p, lc)."""
    mesh, cell_tags, facet_tags = build_robin_mesh(comm, r_s=r_s, L=L, lc=lc,
                                                   order=order)
    data = IncidentPlaneWave(mesh, k, inc, beta=beta, alpha=alpha, r_s=r_s)

    W = dfx.fem.functionspace(mesh, ("Lagrange", p + 2))
    Wv = dfx.fem.functionspace(mesh, ("Lagrange", p + 2, (mesh.geometry.dim,)))
    umW = dfx.fem.Function(W)
    umW.interpolate(mie_callable(mie, k))
    gmW = dfx.fem.Function(Wv)
    gmW.interpolate(mie_grad_callable(mie, k))

    solver = RobinScatteringSolver(mesh, facet_tags, p, data, umW, gmW)
    solver.solve()

    dx = ufl.dx(domain=mesh)
    ds_scat = ufl.Measure("ds", mesh, subdomain_data=facet_tags)(FACET_ID.SCAT)
    uhW = dfx.fem.Function(W)
    uhW.interpolate(solver.u_sc)
    eF = dfx.fem.Function(W)
    eF.x.array[:] = uhW.x.array - umW.x.array

    e = energy_norm(eF, k, mesh, dx, ds_scat, comm)
    h = h_max(mesh, cell_tags, comm)
    solver.destroy()
    return h, e


def eoc(h, e):
    """Estimated order of convergence from a sequence of (h, error) pairs."""
    h, e = np.asarray(h), np.asarray(e)
    p = np.log(e[1:] / e[:-1]) / np.log(h[1:] / h[:-1])
    return float(np.median(p))


def main():
    """Run the convergence check and print PASS/FAIL."""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    p = 2
    k = 4.0
    beta = complex(1.0, 0.0)
    alpha = complex(0.6, 0.0)
    r_s, L, order = 1.0, 2.0, 3
    inc = np.array([1.0, 0.0])
    inc /= np.linalg.norm(inc)
    mie = MieWentzellCylinder(r_s, beta, alpha, inc)

    lcs = [0.40, 0.28, 0.20, 0.14]
    hs, es = [], []
    for lc in lcs:
        h, e = energy_error(p, k, lc, beta, alpha, r_s, L, order, inc, mie, comm)
        hs.append(h)
        es.append(e)
        if rank == 0:
            print(f"  lc={lc:.3f}  h={h:.4e}  energy_err={e:.6e}", flush=True)

    rate = eoc(hs, es)
    ok = rate > p - 0.5
    if rank == 0:
        print(f"\np={p} k={k} beta={beta} alpha={alpha}")
        print(f"observed EOC = {rate:.3f}  (expected ~ {p}; threshold {p-0.5})")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
