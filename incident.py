"""Incident plane wave and scattered-field Wentzell RHS data for the Robin
truncated resolution study.
"""

import numpy as np
import ufl
import dolfinx as dfx
from petsc4py import PETSc


class IncidentPlaneWave:
    """Plane-wave incident field plus the manufactured Wentzell RHS on the obstacle."""

    def __init__(
        self,
        mesh: dfx.mesh.Mesh,
        wave_num: float,
        wave_vec: np.ndarray,
        beta: complex = 1.0 + 0.0j,
        alpha: complex = 0.0 + 0.0j,
        r_s: float = 1.0,
    ):
        """Args:
            beta: Wentzell/Laplace-Beltrami coefficient in the boundary condition.
            alpha: lower-order (reactive) coefficient in the boundary condition.
            r_s: obstacle radius, used to evaluate the boundary Laplacian.
        """
        assert wave_vec.shape == (2,)
        assert np.isclose(np.linalg.norm(wave_vec), 1.0)
        assert r_s > 0

        self.mesh = mesh
        self.wave_vec = np.asarray(wave_vec, dtype=np.float64)
        self.r_s = float(r_s)

        self.wave_num = float(wave_num)
        self.beta = complex(beta)
        self.alpha = complex(alpha)

        self.k = dfx.fem.Constant(mesh, PETSc.ScalarType(self.wave_num))
        self.beta_c = dfx.fem.Constant(mesh, PETSc.ScalarType(self.beta))
        self.alpha_c = dfx.fem.Constant(mesh, PETSc.ScalarType(self.alpha))

    def u_inc_ufl(self, x):
        """Incident plane wave as a UFL expression."""
        d = self.wave_vec
        return ufl.exp(1.0j * self.k * (d[0] * x[0] + d[1] * x[1]))

    def u_inc_np(self, x):
        """Incident plane wave as a numpy callable, for interpolation."""
        d = self.wave_vec
        return np.exp(1.0j * self.wave_num * (d[0] * x[0] + d[1] * x[1]))

    def _wentzell_bracket(self, x):
        """Wentzell operator applied to the incident field on Gamma_s."""
        u = self.u_inc_ufl(x)
        n = ufl.FacetNormal(self.mesh)
        angular = (
            x[1] ** 2 * u.dx(0).dx(0)
            - 2.0 * x[0] * x[1] * u.dx(1).dx(0)
            + x[0] ** 2 * u.dx(1).dx(1)
            - x[0] * u.dx(0)
            - x[1] * u.dx(1)
        )
        return (
            ufl.dot(ufl.grad(u), n)
            + (self.beta_c / self.k) * (angular / (self.r_s**2))
            - self.alpha_c * self.k * u
        )

    def g_scat_ufl(self, x):
        """Scattered-field RHS on the obstacle, minus the Wentzell bracket."""
        return -self._wentzell_bracket(x)

    def set_wave_num(self, wave_num: float) -> None:
        """Update the wave number in place."""
        self.wave_num = float(wave_num)
        self.k.value = PETSc.ScalarType(self.wave_num)

    def set_beta(self, beta: complex) -> None:
        """Update the Wentzell coefficient in place."""
        self.beta = complex(beta)
        self.beta_c.value = PETSc.ScalarType(self.beta)

    def set_alpha(self, alpha: complex) -> None:
        """Update the lower-order coefficient in place."""
        self.alpha = complex(alpha)
        self.alpha_c.value = PETSc.ScalarType(self.alpha)
