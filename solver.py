"""Scattered-field Helmholtz solver: Wentzell obstacle BC + first-order Robin
(absorbing) truncation, manufactured so the exact Mie field is the solution.
"""

import dolfinx as dfx
from dolfinx.fem import petsc as dfxpetsc
from petsc4py import PETSc
import ufl

from mesh import FACET_ID


class RobinScatteringSolver:
    """FEM solver for the Robin-truncated Wentzell scattering problem."""

    def __init__(
        self,
        mesh,
        facet_tags,
        pol_deg: int,
        data,
        umie_W: dfx.fem.Function,
        gmie_W: dfx.fem.Function,
    ):
        """Args:
            data: IncidentPlaneWave supplying k, beta, alpha and the RHS data.
            umie_W: exact Mie scattered field, interpolated at the current k.
            gmie_W: gradient of umie_W, interpolated at the current k.
        """
        self.mesh = mesh
        self.pol_deg = pol_deg
        self.data = data

        self.V = dfx.fem.functionspace(mesh, ("Lagrange", pol_deg))
        self.u_sc = dfx.fem.Function(self.V)
        self.u_sc.name = "u_sc"

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        x = ufl.SpatialCoordinate(mesh)

        dx = ufl.dx(domain=mesh)
        ds = ufl.Measure("ds", mesh, subdomain_data=facet_tags)

        n = ufl.FacetNormal(mesh)
        P = ufl.Identity(mesh.geometry.dim) - ufl.outer(n, n)
        grad_t_u = ufl.dot(P, ufl.grad(u))
        grad_t_v = ufl.dot(P, ufl.grad(v))

        k = data.k
        beta = data.beta_c
        alpha = data.alpha_c

        a_dom = (
            ufl.inner(ufl.grad(u), ufl.grad(v)) - k**2 * ufl.inner(u, v)
        ) * dx

        a_scat = (
            -1.0 * (beta / k) * ufl.inner(grad_t_u, grad_t_v)
            - alpha * k * ufl.inner(u, v)
        ) * ds(FACET_ID.SCAT)

        a_robin = -1.0j * k * ufl.inner(u, v) * ds(FACET_ID.OUTER)

        self.a_form = dfx.fem.form(a_dom + a_scat + a_robin)

        g_robin = ufl.dot(gmie_W, n) - 1.0j * k * umie_W
        self.L_form = dfx.fem.form(
            ufl.inner(data.g_scat_ufl(x), v) * ds(FACET_ID.SCAT)
            + ufl.inner(g_robin, v) * ds(FACET_ID.OUTER)
        )

        self.A = dfxpetsc.create_matrix(self.a_form)
        self.rhs_vector = dfxpetsc.create_vector(self.V)

        self.opts = PETSc.Options()
        self.opts["ksp_type"] = "preonly"
        self.opts["pc_type"] = "lu"
        self.opts["pc_factor_mat_solver_type"] = "mumps"
        self.solver = PETSc.KSP().create(mesh.comm)
        self.solver.setFromOptions()

    def solve(self) -> dfx.fem.Function:
        """Assemble and solve at the current wave number."""
        self.A.zeroEntries()
        dfxpetsc.assemble_matrix(self.A, self.a_form)
        self.A.assemble()

        with self.rhs_vector.localForm() as loc_b:
            loc_b.set(0)
        dfxpetsc.assemble_vector(self.rhs_vector, self.L_form)
        self.rhs_vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )
        self.rhs_vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        self.solver.setOperators(self.A)
        self.solver.solve(self.rhs_vector, self.u_sc.x.petsc_vec)
        self.u_sc.x.scatter_forward()
        return self.u_sc

    def destroy(self) -> None:
        """Release the KSP solver and PETSc matrix/vector."""
        self.solver.destroy()
        self.A.destroy()
        self.rhs_vector.destroy()
