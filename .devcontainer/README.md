# Development container

VSCode development container for this project, built on the official
[DOLFINx](https://hub.docker.com/r/dolfinx/dolfinx) image.

Two things are worth knowing:

* The base image is pinned by **manifest digest** (DOLFINx v0.11.0), not by the
  `stable` tag. `stable` moves with every release and this code targets the 0.11
  API, so an unpinned build would eventually break.
* Complex PETSc scalars are enabled by default via `ENV` in the Dockerfile.
  Inside this container you do **not** need to `source dolfinx-complex-mode`.

`dolfinx`, `mpi4py`, `petsc4py`, `ufl` and `gmsh` come from the base image;
`requirements.txt` only pins the pure-Python additions (numpy, scipy,
matplotlib).

See [../README.md](../README.md) for how to build the container and reproduce
the results.
