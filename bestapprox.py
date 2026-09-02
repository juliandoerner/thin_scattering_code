"""Energy-norm best approximation of the exact field onto the FEM space."""

import dolfinx as dfx
from dolfinx.fem import petsc as dfxpetsc
from petsc4py import PETSc
import ufl

from mesh import FACET_ID


class EnergyBestApproximation:
    """Galerkin projection of a given field onto P^pol_deg in the energy inner product."""

    def __init__(
        self,
        mesh,
        facet_tags,
        pol_deg: int,
        k: float,
        u_ex,
        grad_u_ex=None,
    ):
        """Args:
            grad_u_ex: gradient of u_ex; defaults to ufl.grad(u_ex) if not given.
        """
        self.mesh = mesh
        self.pol_deg = pol_deg

        self.V = dfx.fem.functionspace(mesh, ("Lagrange", pol_deg))
        self.u_best = dfx.fem.Function(self.V)
        self.u_best.name = "u_best"

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        dx = ufl.dx(domain=mesh)
        ds_scat = ufl.Measure("ds", mesh, subdomain_data=facet_tags)(FACET_ID.SCAT)

        n = ufl.FacetNormal(mesh)
        P = ufl.Identity(mesh.geometry.dim) - ufl.outer(n, n)

        self.k = dfx.fem.Constant(mesh, PETSc.ScalarType(k))
        kc = self.k

        if grad_u_ex is None:
            grad_u_ex = ufl.grad(u_ex)

        self.a_form = dfx.fem.form(
            kc**2 * ufl.inner(u, v) * dx
            + ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
            + (1.0 / kc) * ufl.inner(ufl.dot(P, ufl.grad(u)),
                                     ufl.dot(P, ufl.grad(v))) * ds_scat
            + kc * ufl.inner(u, v) * ds_scat
        )
        self.L_form = dfx.fem.form(
            kc**2 * ufl.inner(u_ex, v) * dx
            + ufl.inner(grad_u_ex, ufl.grad(v)) * dx
            + (1.0 / kc) * ufl.inner(ufl.dot(P, grad_u_ex),
                                     ufl.dot(P, ufl.grad(v))) * ds_scat
            + kc * ufl.inner(u_ex, v) * ds_scat
        )

        self.A = dfxpetsc.create_matrix(self.a_form)
        self.rhs_vector = dfxpetsc.create_vector(self.V)

        self.solver = PETSc.KSP().create(mesh.comm)
        self.solver.setType("preonly")
        pc = self.solver.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")

    def solve(self) -> dfx.fem.Function:
        """Assemble and solve the projection system."""
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
        self.solver.solve(self.rhs_vector, self.u_best.x.petsc_vec)
        self.u_best.x.scatter_forward()
        return self.u_best

    def destroy(self) -> None:
        """Release the KSP solver and PETSc matrix/vector."""
        self.solver.destroy()
        self.A.destroy()
        self.rhs_vector.destroy()
