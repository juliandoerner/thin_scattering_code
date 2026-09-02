#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Critical-resolution vs wave number for a Robin-truncated, manufactured Mie
scattering problem, for polynomial degrees p = 1, 2, 3, 4.

Usage (complex PETSc required)::

    source dolfinx-complex-mode
    mpirun -n N python run_resolution.py -p 1 2 3 4 -k 6 8 10 12 14 16 \
        -b 1.0 -a 1.0 --thresh 1.5 -o output
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx as dfx
import ufl

from mesh import build_robin_mesh, FACET_ID
from incident import IncidentPlaneWave
from solver import RobinScatteringSolver
from bestapprox import EnergyBestApproximation
from mie import MieWentzellCylinder
from energy import (
    energy_norm, h_max, mie_callable, mie_grad_callable, find_crossing,
)

assert np.dtype(PETSc.ScalarType).kind == "c", "activate: source dolfinx-complex-mode"

def _complex(s):
    """Parse a command-line argument into a complex number."""
    return complex(s)


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Robin/Mie critical-resolution vs k")
    p.add_argument("-o", "--output_dir", type=Path, default=Path("output"))
    p.add_argument("-p", "--pol_degs", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("-k", "--wave_nums", type=float, nargs="+",
                   default=[6, 8, 10, 12, 14, 16])
    p.add_argument("-b", "--beta", type=_complex, default=complex(1.0, 0.0))
    p.add_argument("-a", "--alpha", type=_complex, default=complex(1.0, 0.0))
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--inc", type=str, default="1,0")
    p.add_argument("-L", "--L", type=float, default=3.0)
    p.add_argument("--thresh", type=float, default=1.5,
                   help="'close enough': err_fem <= thresh * err_best")
    p.add_argument("--kh_min", type=float, default=0.25)
    p.add_argument("--kh_max", type=float, default=3.0)
    p.add_argument("--n_kh", type=int, default=12)
    p.add_argument("--order", type=int, default=3,
                   help="geometric mesh order (gmsh supports up to 3)")
    p.add_argument("--max_cells", type=int, default=3_000_000)
    return p.parse_args()


def solve_one(k, lc, p, args, inc, mie, comm):
    """Solve one (k, lc, p) combination and return the errors, or None if too large."""
    box = 2.0 * args.L
    if 2.0 * box**2 / lc**2 > args.max_cells:
        return None

    mesh, cell_tags, facet_tags = build_robin_mesh(
        comm, r_s=args.radius, L=args.L, lc=lc, order=args.order)

    data = IncidentPlaneWave(mesh, k, inc, beta=args.beta, alpha=args.alpha,
                             r_s=args.radius)

    W = dfx.fem.functionspace(mesh, ("Lagrange", p + 2))
    Wv = dfx.fem.functionspace(mesh, ("Lagrange", p + 2, (mesh.geometry.dim,)))
    umW = dfx.fem.Function(W)
    umW.interpolate(mie_callable(mie, k))
    gmW = dfx.fem.Function(Wv)
    gmW.interpolate(mie_grad_callable(mie, k))

    solver = RobinScatteringSolver(mesh, facet_tags, p, data, umW, gmW)
    solver.solve()
    uhW = dfx.fem.Function(W)
    uhW.interpolate(solver.u_sc)
    vh = dfx.fem.Function(solver.V)
    vh.interpolate(mie_callable(mie, k))
    viW = dfx.fem.Function(W)
    viW.interpolate(vh)
    solver.destroy()

    best = EnergyBestApproximation(mesh, facet_tags, p, k, umW)
    best.solve()
    ubW = dfx.fem.Function(W)
    ubW.interpolate(best.u_best)
    best.destroy()

    dx = ufl.dx(domain=mesh)
    ds_scat = ufl.Measure("ds", mesh, subdomain_data=facet_tags)(FACET_ID.SCAT)

    eF = dfx.fem.Function(W)
    eF.x.array[:] = uhW.x.array - umW.x.array
    eB = dfx.fem.Function(W)
    eB.x.array[:] = ubW.x.array - umW.x.array
    eI = dfx.fem.Function(W)
    eI.x.array[:] = viW.x.array - umW.x.array

    norm_u = energy_norm(umW, k, mesh, dx, ds_scat, comm)
    e_fem = energy_norm(eF, k, mesh, dx, ds_scat, comm)
    e_best = energy_norm(eB, k, mesh, dx, ds_scat, comm)
    e_interp = energy_norm(eI, k, mesh, dx, ds_scat, comm)
    h = h_max(mesh, cell_tags, comm)

    del mesh, cell_tags, facet_tags, data, solver, best, W, Wv
    ratio = e_fem / e_best if e_best > 0 else np.nan
    return h, ratio, e_best / norm_u, e_fem / norm_u, e_interp / norm_u


def run_degree(p, args, inc, mie, comm, rank):
    """Sweep all (k, kh) for one polynomial degree; return (crit_rows, sweep_rows)."""
    ks = sorted(args.wave_nums)
    kh_grid = np.geomspace(args.kh_max, args.kh_min, args.n_kh)
    sweep_rows, crit_rows = [], []

    for k in ks:
        khs, ratios = [], []
        for kh_t in kh_grid:
            lc = round(kh_t / k, 6)
            comm.Barrier()
            ts = time.time()
            res = solve_one(k, lc, p, args, inc, mie, comm)
            if res is None:
                continue
            h, ratio, rel_best, rel_fem, rel_interp = res
            kh = k * h
            khs.append(kh)
            ratios.append(ratio)
            if rank == 0:
                sweep_rows.append((k, kh, ratio, rel_best, rel_fem, rel_interp))
                warn = "  [WARN best > interp]" if rel_best > 1.001 * rel_interp else ""
                print(f"  p={p} k={k:5.1f} lc={lc:.4f} kh={kh:6.3f} "
                      f"ratio={ratio:7.3f} (rb={rel_best:.2e} rf={rel_fem:.2e} "
                      f"ri={rel_interp:.2e}) {time.time()-ts:.1f}s{warn}", flush=True)
        kh_crit, status = find_crossing(khs, ratios, args.thresh)
        h_crit = kh_crit / k
        if rank == 0:
            crit_rows.append((k, kh_crit, h_crit, status))
            print(f"  --> p={p} k={k:5.1f} kh_crit={kh_crit:.4f} "
                  f"h_crit={h_crit:.4e} ({status})", flush=True)
    return crit_rows, sweep_rows


def main():
    """Run the full sweep over polynomial degrees and write the output CSVs."""
    args = parse_args()
    t0 = time.time()
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size

    inc = np.array([float(v) for v in args.inc.split(",")])
    inc /= np.linalg.norm(inc)
    mie = MieWentzellCylinder(args.radius, args.beta, args.alpha, inc)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"pol_degs={args.pol_degs} beta={args.beta} alpha={args.alpha} "
              f"rad={args.radius} L={args.L} thresh={args.thresh}", flush=True)

    excluded = {}
    for p in args.pol_degs:
        crit_rows, sweep_rows = run_degree(p, args, inc, mie, comm, rank)
        if rank != 0:
            continue
        ok_rows = [(r[0], r[1], r[2]) for r in crit_rows if r[3] == "ok"]
        skipped = [(r[0], r[3]) for r in crit_rows if r[3] != "ok"]
        excluded[p] = skipped
        if skipped:
            print(f"  [p={p}] excluded {len(skipped)} non-ok k from critical CSV: "
                  f"{', '.join(f'{k:g}({s})' for k, s in skipped)}", flush=True)
        crit = np.array(ok_rows, dtype=float)
        sweep = np.array(sweep_rows, dtype=float)
        np.savetxt(args.output_dir / f"critical_p{p}.csv", crit, delimiter=",",
                   header="k,kh_crit,h_crit", comments="")
        np.savetxt(args.output_dir / f"sweep_p{p}.csv", sweep, delimiter=",",
                   header="k,kh,ratio,rel_best,rel_fem,rel_interp", comments="")

    if rank == 0:
        with open(args.output_dir / "study_info.txt", "w") as f:
            f.write("Robin/Mie critical-resolution vs k\n" + "=" * 60 + "\n\n")
            f.write(f"wall time (s): {time.time()-t0:.2f}\n"
                    f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"ranks: {size}\ncommand: {' '.join(sys.argv)}\n\n")
            f.write(f"pol_degs={args.pol_degs} beta={args.beta} "
                    f"alpha={args.alpha} a={args.radius} L={args.L}\n")
            f.write(f"thresh={args.thresh} kh in [{args.kh_min},{args.kh_max}] "
                    f"n_kh={args.n_kh} order={args.order}\n")
            f.write("err_best = energy-inner-product Galerkin projection of the "
                    "Mie field onto V_h (bestapprox.py); rel_interp is the nodal "
                    "interpolant, reported as a diagnostic upper bound.\n")
            for p, sk in excluded.items():
                if sk:
                    f.write(f"p={p} excluded (crossing outside kh window): "
                            f"{', '.join(f'{k:g}({s})' for k, s in sk)}\n")
        print("=== FINISHED ===", flush=True)


if __name__ == "__main__":
    main()
