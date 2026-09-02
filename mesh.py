"""Gmsh geometry for the Robin-truncated scattering resolution study."""

from enum import IntEnum

import dolfinx as dfx
import gmsh
from mpi4py import MPI


class CELL_ID(IntEnum):
    PHYS = 1


class FACET_ID(IntEnum):
    SCAT = 10
    OUTER = 20


def _classify_geometry(model, r_s: float, L: float):
    """Classify surfaces and curves by bounding box into physical/obstacle/scat/outer."""
    atol = 0.05 * min(r_s, L)

    phys_surfs, obstacle_surfs = [], []
    for dim, tag in model.occ.getEntities(2):
        xmin, ymin, _, xmax, ymax, _ = model.occ.getBoundingBox(dim, tag)
        reach = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
        if abs(reach - L) < atol:
            phys_surfs.append(tag)
        elif reach < r_s + atol:
            obstacle_surfs.append(tag)
        else:  # pragma: no cover - should not happen for valid params
            raise RuntimeError(
                f"Unclassified surface {tag}: bbox reach {reach} (L={L})"
            )

    scat_curves, outer_curves = [], []
    for dim, tag in model.occ.getEntities(1):
        xmin, ymin, _, xmax, ymax, _ = model.occ.getBoundingBox(dim, tag)
        reach = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
        if abs(reach - L) < atol:
            outer_curves.append(tag)
        elif reach < r_s + atol:
            scat_curves.append(tag)

    return phys_surfs, obstacle_surfs, scat_curves, outer_curves


def build_robin_mesh(
    comm: MPI.Comm,
    r_s: float = 1.0,
    L: float = 3.0,
    lc: float = 0.2,
    order: int = 3,
):
    """Build the mesh: obstacle hole in a square air region, no PML.

    Args:
        r_s: obstacle radius.
        L: half-width of the square domain [-L, L]^2.
        lc: target mesh element size.
        order: geometric order of the mesh elements (gmsh supports up to 3).
    """
    assert r_s > 0 and L > r_s and lc > 0

    rank = comm.rank

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    model = gmsh.model()
    model.add("robin_scattering")

    if rank == 0:
        air = model.occ.addRectangle(-L, -L, 0.0, 2 * L, 2 * L)
        disk = model.occ.addDisk(0.0, 0.0, 0.0, r_s, r_s)

        model.occ.fragment([(2, air)], [(2, disk)])
        model.occ.synchronize()

        phys_s, obst_s, scat_c, outer_c = _classify_geometry(model, r_s, L)

        model.occ.remove([(2, t) for t in obst_s], recursive=False)
        model.occ.synchronize()

        model.addPhysicalGroup(2, phys_s, CELL_ID.PHYS)
        model.addPhysicalGroup(1, scat_c, FACET_ID.SCAT)
        model.addPhysicalGroup(1, outer_c, FACET_ID.OUTER)

        gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
        gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)

        model.mesh.generate(2)
        model.mesh.setOrder(order)
        model.mesh.optimize("Netgen")

    mesh_data = dfx.io.gmsh.model_to_mesh(model, comm, rank=0, gdim=2)

    gmsh.clear()
    gmsh.finalize()

    mesh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags
    cell_tags.name = "cell_tags"
    facet_tags.name = "facet_tags"

    return mesh, cell_tags, facet_tags
