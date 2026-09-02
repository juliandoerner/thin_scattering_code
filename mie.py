r"""Exact (Mie / separation-of-variables) reference for plane-wave scattering by
a circular obstacle with a generalised Wentzell boundary condition.
"""

import numpy as np
from scipy import special as sp


class MieWentzellCylinder:
    """Analytic Mie solution for a Wentzell-coated cylinder."""

    def __init__(self, a: float, beta: complex, alpha: complex = 0.0,
                 inc_dir=(1.0, 0.0), n_extra: int = 25):
        """Args:
            beta: Wentzell (Laplace-Beltrami) coefficient.
            alpha: lower-order reactive coefficient.
            n_extra: number of Fourier modes kept beyond ceil(k*a) when
                truncating the series.
        """
        self.a = float(a)
        self.beta = complex(beta)
        self.alpha = complex(alpha)
        d = np.asarray(inc_dir, dtype=np.float64)
        d /= np.linalg.norm(d)
        self.alpha_inc = float(np.arctan2(d[1], d[0]))
        self.n_extra = int(n_extra)

    def _modes(self, k: float, N=None):
        """Return the range of Fourier modes to sum over."""
        if N is None:
            N = int(np.ceil(k * self.a)) + self.n_extra
        return np.arange(-N, N + 1)

    def coefficients(self, k: float, N=None):
        """Return (n_array, a_n, c_inc_n) for wave number k.

        Args:
            N: cap on the number of Fourier modes; leave None to pick it
                automatically from k (see _modes).
        """
        n = self._modes(k, N)
        z = k * self.a
        Jn = sp.jv(n, z)
        Jnp = sp.jvp(n, z, 1)
        Hn = sp.hankel1(n, z)
        Hnp = sp.h1vp(n, z, 1)

        w = self.beta * n**2 / (k * self.a**2) + self.alpha * k
        c_inc = (1j ** n) * np.exp(-1j * n * self.alpha_inc)

        num = k * Jnp + w * Jn
        den = k * Hnp + w * Hn
        a_n = -c_inc * num / den
        return n, a_n, c_inc

    def _polar(self, points):
        """Convert Cartesian points to polar coordinates."""
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        r = np.hypot(p[:, 0], p[:, 1])
        th = np.arctan2(p[:, 1], p[:, 0])
        return r, th

    def u_scattered(self, k: float, points):
        """Evaluate the scattered field at the given points."""
        n, a_n, _ = self.coefficients(k)
        r, th = self._polar(points)
        Hr = sp.hankel1(n[None, :], k * r[:, None])
        ph = np.exp(1j * np.outer(th, n))
        return np.sum(a_n[None, :] * Hr * ph, axis=1)

    def grad_scattered(self, k: float, points):
        """Evaluate the Cartesian gradient of the scattered field, shape (N, 2)."""
        n, a_n, _ = self.coefficients(k)
        r, th = self._polar(points)
        kr = k * r[:, None]
        ph = np.exp(1j * np.outer(th, n))
        Hr = sp.hankel1(n[None, :], kr)
        Hrp = sp.h1vp(n[None, :], kr, 1)

        du_dr = np.sum(a_n[None, :] * k * Hrp * ph, axis=1)
        du_dt_over_r = np.sum(
            a_n[None, :] * Hr * (1j * n[None, :]) * ph, axis=1
        ) / r
        c, s = np.cos(th), np.sin(th)
        gx = du_dr * c - du_dt_over_r * s
        gy = du_dr * s + du_dt_over_r * c
        return np.column_stack([gx, gy])

    def u_incident(self, k: float, points):
        """Evaluate the incident plane wave at the given points."""
        n, _, c_inc = self.coefficients(k)
        r, th = self._polar(points)
        Jr = sp.jv(n[None, :], k * r[:, None])
        ph = np.exp(1j * np.outer(th, n))
        return np.sum(c_inc[None, :] * Jr * ph, axis=1)

    def u_total(self, k: float, points):
        """Evaluate the total field (incident + scattered) at the given points."""
        return self.u_incident(k, points) + self.u_scattered(k, points)
