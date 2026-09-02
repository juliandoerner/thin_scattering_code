# DOLFINx v0.11.0, pinned by manifest digest so the environment is reproducible.
# The `stable` tag moves with each release and this code targets the 0.11 API.
FROM dolfinx/dolfinx@sha256:58b27e84a2f26b98ce2d9ccc537b0ee6a59e2fcfdf386626d5ed9ddf43425ece

# Every entry point in this project requires complex-valued PETSc scalars, but
# the image defaults to the real build.  These are the paths `dolfinx-complex-mode`
# would export, written out in full: the complex tree is placed ahead of the real
# one, and /usr/local/lib (which provides gmsh) is kept on PYTHONPATH.  Baking it
# in here rather than sourcing per shell means mpirun, the VSCode debugger and a
# plain `python3` all get complex mode automatically.
ENV PETSC_ARCH=linux-gnu-complex128-32
ENV PYTHONPATH=/usr/local/dolfinx-complex/lib/python3.12/dist-packages:/usr/local/dolfinx-real/lib/python3.12/dist-packages:/usr/local/lib
ENV LD_LIBRARY_PATH=/usr/local/dolfinx-complex/lib:/usr/local/dolfinx-real/lib
ENV PKG_CONFIG_PATH=/usr/local/dolfinx-complex/lib/pkgconfig:/usr/local/dolfinx-real/lib/pkgconfig
ENV CMAKE_PREFIX_PATH=/usr/local/dolfinx-complex/lib/cmake:/usr/local/dolfinx-real/lib/cmake

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
