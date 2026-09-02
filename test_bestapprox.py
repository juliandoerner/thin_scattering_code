#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation for the energy-norm best approximation (bestapprox.py).

Checks Galerkin orthogonality (Pythagoras) and that the projection beats the
nodal interpolant, on the curved obstacle mesh used by the resolution study.

Run (complex PETSc required)::

    source dolfinx-complex-mode
    python test_bestapprox.py            # serial
    mpirun -n 2 python test_bestapprox.py
"""

import sys

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx as dfx
import ufl

from mesh import build_robin_mesh, FACET_ID
from bestapprox import EnergyBestApproximation
from mie import MieWentzellCylinder
from energy import energy_norm, mie_callable

assert np.dtype(PETSc.ScalarType).kind == "c", "activate: source dolfinx-complex-mode"


def check_one(p, k, lc, r_s, L, order, mie, comm):
    """Return (e_best, e_interp, pythagoras_residual) for one (p, lc)."""
    mesh, _, facet_tags = build_robin_mesh(comm, r_s=r_s, L=L, lc=lc, order=order)

    W = dfx.fem.functionspace(mesh, ("Lagrange", p + 2))
    umW = dfx.fem.Function(W)
    umW.interpolate(mie_callable(mie, k))

    best = EnergyBestApproximation(mesh, facet_tags, p, k, umW)
    best.solve()

    vh = dfx.fem.Function(best.V)
    vh.interpolate(mie_callable(mie, k))

    ubW = dfx.fem.Function(W)
    ubW.interpolate(best.u_best)
    viW = dfx.fem.Function(W)
    viW.interpolate(vh)

    dx = ufl.dx(domain=mesh)
    ds_scat = ufl.Measure("ds", mesh, subdomain_data=facet_tags)(FACET_ID.SCAT)

    def enorm(a, b):
        """Energy norm of the difference between two functions."""
        w = dfx.fem.Function(W)
        w.x.array[:] = a.x.array - b.x.array
        return energy_norm(w, k, mesh, dx, ds_scat, comm)

    e_best = enorm(ubW, umW)
    e_interp = enorm(viW, umW)
    d = enorm(ubW, viW)

    best.destroy()
    res = abs(e_interp**2 - (e_best**2 + d**2)) / e_interp**2
    return e_best, e_interp, res


def main():
    """Run the check on a couple of mesh sizes and print PASS/FAIL."""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    p, k = 2, 4.0
    beta, alpha = complex(1.0, 0.0), complex(0.6, 0.0)
    r_s, L, order = 1.0, 2.0, 3
    inc = np.array([1.0, 0.0])
    inc /= np.linalg.norm(inc)
    mie = MieWentzellCylinder(r_s, beta, alpha, inc)

    tol_pyth, ok = 1e-6, True
    for lc in [0.40, 0.25]:
        e_best, e_interp, res = check_one(p, k, lc, r_s, L, order, mie, comm)
        good = (res < tol_pyth) and (e_best <= e_interp)
        ok = ok and good
        if rank == 0:
            print(f"  lc={lc:.3f}  ||u-u_B||={e_best:.6e}  "
                  f"||u-I_h u||={e_interp:.6e}  (gain {e_interp/e_best:.4f})  "
                  f"pythagoras residual={res:.2e}", flush=True)

    if rank == 0:
        print(f"\np={p} k={k} beta={beta} alpha={alpha}")
        print(f"checks: Pythagoras residual < {tol_pyth:g} and ||u-u_B|| <= ||u-I_h u||")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
