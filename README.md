# Critical resolution for Wentzell-coated scattering

This software is published for reproducibility, following the guidelines for
safeguarding good research practice of the Deutsche Forschungsgemeinschaft
(DFG). It contains the source code, a version-pinned environment and the
parameter sets needed to regenerate the reported results.

Numerical study of the resolution condition for a finite element discretisation
of the Helmholtz equation, for a plane wave scattered by a circular obstacle
with a generalised Wentzell boundary condition.

For each wave number `k` we sweep the mesh size and locate the coarsest
resolution `kh` at which the FEM error is still within a factor `--thresh` of
the best approximation in the same FEM space. On log-log axes the critical mesh
size against `k` tests

```
k (k h_crit)^p = const      =>      h_crit ~ k^{-(p+1)/p}
```

so the fitted slope should approach -2.00 for `p=1`, -1.50 for `p=2` and -1.33
for `p=3`.

The exterior Robin data is manufactured from the exact Mie field, so the Mie
field solves the continuous problem exactly and the measured error is pure
discretisation error. The reference is the true best approximation, i.e. the
Galerkin projection of the exact field onto the FEM space in the energy inner
product (`bestapprox.py`). A nodal interpolant would only be an upper bound and
would under-report the loss of quasi-optimality; it is still computed and
reported alongside as a diagnostic.

## Environment

Everything runs in a container built on the official DOLFINx image, pinned by
manifest digest to v0.11.0:

- DOLFINx 0.11.0, PETSc/petsc4py 3.25.1 (complex128)
- gmsh 4.15.2
- numpy 2.4.6, scipy 1.17.1, matplotlib 3.10.9
- Ubuntu 24.04, Python 3.12.3

The code requires complex PETSc scalars and asserts this at import. The
container enables complex mode by default, so nothing has to be sourced.

In VSCode, open the repository and choose "Reopen in Container"; the definition
is in [.devcontainer/](.devcontainer/).

Without VSCode, use the image directly:

```bash
docker run -it --rm -v "$PWD":/root/shared -w /root/shared \
    dolfinx/dolfinx@sha256:58b27e84a2f26b98ce2d9ccc537b0ee6a59e2fcfdf386626d5ed9ddf43425ece \
    bash

# inside the container:
source dolfinx-complex-mode
pip install -r .devcontainer/requirements.txt
```

That path uses the stock image, so complex mode has to be sourced by hand. The
dev container build bakes it in instead.

## Checking the installation

Run the two validation scripts first. Both print `RESULT: PASS` or
`RESULT: FAIL` and exit non-zero on failure.

```bash
python test_robin.py         # energy error converges at O(h^p)
python test_bestapprox.py    # projection is orthogonal and beats the interpolant
```

`test_robin.py` pins the signs in the pipeline. If the Wentzell term, the Robin
term or one of the manufactured right-hand sides had a wrong sign, the Mie field
would no longer be the exact solution, the error would plateau and the observed
order of convergence would fail its assertion. `test_bestapprox.py` checks
Galerkin orthogonality via `||u - I_h u||^2 = ||u - u_B||^2 + ||u_B - I_h u||^2`.

Both take a few minutes serially and also run under `mpirun -n 2`.

## Reproducing the results

```bash
./run_all.sh            # 8 MPI ranks
./run_all.sh 4          # or pick the rank count
```

This runs `p=1` and `p=2` and renders the figures. The `p=3` study needs much
more memory and is opt-in:

```bash
RUN_HIGH_P=1 ./run_all.sh 8
```

The parameter sets are in [run_all.sh](run_all.sh):

| p | k               | kh window  | n_kh | L   | max cells |
|---|-----------------|------------|------|-----|-----------|
| 1 | 6..14 step 1    | 0.25 - 3.0 | 20   | 2.0 | 500000    |
| 2 | 10..26 step 2   | 0.5 - 3.0  | 20   | 2.0 | 5000000   |
| 3 | 60..110 step 5  | 1.0 - 3.5  | 20   | 2.5 | 8000000   |

All three use `--thresh 1.2`, so "close enough" means the FEM energy error stays
within 20% of the best approximation. Runs exceeding `--max_cells` are skipped.

A single configuration:

```bash
mpirun -n 8 python run_resolution.py -p 2 -k 10 12 14 16 -o my_output --thresh 1.2
python make_figures.py -o my_output
```

See `python run_resolution.py --help` for the remaining arguments (obstacle
radius, incident direction, the Wentzell coefficients `beta` and `alpha`, and
the geometric mesh order).

## Outputs

Written to `output/`, or wherever `-o` points, one set per degree:

- `critical_p{p}.csv`: `k, kh_crit, h_crit`, the fitted quantities
- `sweep_p{p}.csv`: `k, kh, ratio, rel_best, rel_fem, rel_interp`, the full sweep
- `resolution_p{p}.png`: `h_crit` vs `k` with the slope fit, `kh_crit` vs `k`,
  and the ratio sweeps
- `study_info.txt`: parameters, rank count, wall time, invoking command

`make_figures.py` prints the fitted slope next to the theoretical one, which is
the number of interest:

```
[ok] p=2: fitted slope -1.497 (theory -1.500) -> resolution_p2.png
```

Only wave numbers whose crossing lies inside the swept `kh` window reach
`critical_p{p}.csv`. If the ratio never crosses the threshold (`resolved_all`)
or sits above it throughout (`polluted_all`), the crossing is a window artefact
rather than a measurement, so it is dropped from the fit and listed in
`study_info.txt`. Widen `--kh_min` / `--kh_max` if many are dropped.

Reference CSVs from our own runs are in [reference_output/](reference_output/)
for comparison. Exact agreement should not be expected: MUMPS reorders
floating-point reductions with the rank count, so `h_crit` differs in the
trailing digits between runs. The fitted slopes are the reproducible quantity.

## Repository layout

- `run_resolution.py`: driver, sweeps `(p, k, kh)` and writes the CSVs
- `run_all.sh`: the parameter sets used for the reported results
- `make_figures.py`: renders the figures from the CSVs
- `mesh.py`: gmsh geometry, obstacle hole in a square air region
- `incident.py`: incident plane wave and the Wentzell right-hand side
- `solver.py`: Helmholtz solver, Wentzell obstacle plus Robin truncation
- `bestapprox.py`: energy-norm best approximation
- `mie.py`: exact Mie series reference including the analytic gradient
- `energy.py`: energy norm, mesh size and crossing helpers
- `test_robin.py`, `test_bestapprox.py`: validation
