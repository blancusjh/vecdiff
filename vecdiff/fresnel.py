import numpy as np

from .CartesianSurfaces import CartesianSurface
from .geometry import ray_geometry, ray_geometry_from_rho


# === PARAXIAL FRESNEL COEFFICIENTS ===

class FresnelOvoidParax:
    '''
    Provides Fresnel Coeffients for Ovoids defined by its physical parameters in
    the paraxial aproximation.
    '''
    def __init__(self, n0, ni, z0, zi):
        self.n0 = n0
        self.ni = ni
        self.z0 = z0
        self.zi = zi

        self.gamma_0 = 2 * n0 / (n0 + ni)
        self.xi = (
            n0 * (z0 - zi) ** 2
            / (z0**2 * zi**2 * (n0**2 - ni**2))
        )

    def t_s(self, r):
        # Expanding the exact coefficients to O(r^2) gives +ni*xi on the s
        # channel and +n0*xi on the p channel -- both with a plus sign.  The
        # minus here was the documented bug; it flipped t_+ against t_-.
        # Checked against the exact coefficients at r -> 0.
        return self.gamma_0 + self.ni * self.xi * r**2

    def t_p(self, r):
        return self.gamma_0 + self.n0 * self.xi * r**2

    def coeffs(self, r):
        return self.t_s(r), self.t_p(r)



# Exact Fresnel Coefficeints for the Ovoid
class FresnelOvoid:
    """Exact dielectric Fresnel coefficients on the stigmatic oval.

    All public methods take ``r``, the *cylindrical* radius of the surface
    point -- the physical aperture coordinate.  The surface's own parameter
    ``rho`` (the spherical radius from the vertex) is derived internally via
    :meth:`CartesianSurface.rho_from_r`.  The two are not interchangeable: for
    ``z0 = -10, zi = 6`` grazing incidence sits at ``r = 2.397`` but at
    ``rho = 3.758``.
    """

    def __init__(self, ovoid=None, n0=None, z0=None, ni=None, zi=None):
        self.ovoid = ovoid if ovoid is not None else CartesianSurface(
            n0=n0, z0=z0, ni=ni, zi=zi
        )

    # -- geometry --------------------------------------------------- #

    def geometry(self, r):
        """Return the :class:`~vecdiff.geometry.RayGeometry` at pupil radius ``r``."""
        return ray_geometry(self.ovoid, r)

    def _cosines(self, rho):
        """Return ``(cos theta_0, cos theta_i)`` from the surface parameter ``rho``.

        Kept for probes that sweep the surface parametrisation directly.  Use
        :meth:`geometry` when working from the aperture coordinate.
        """
        geom = ray_geometry_from_rho(self.ovoid, rho)
        return geom.cos_t0, geom.cos_ti

    # -- transmission ----------------------------------------------- #

    @staticmethod
    def _ts(n0, ni, cos_t0, cos_ti):
        """Eq. (37)."""
        return 2.0 * n0 * cos_t0 / (n0 * cos_t0 + ni * cos_ti)

    @staticmethod
    def _tp(n0, ni, cos_t0, cos_ti):
        """Eq. (38)."""
        return 2.0 * n0 * cos_t0 / (ni * cos_t0 + n0 * cos_ti)

    @staticmethod
    def _rs(n0, ni, cos_t0, cos_ti):
        return (n0 * cos_t0 - ni * cos_ti) / (n0 * cos_t0 + ni * cos_ti)

    @staticmethod
    def _rp(n0, ni, cos_t0, cos_ti):
        return (ni * cos_t0 - n0 * cos_ti) / (ni * cos_t0 + n0 * cos_ti)

    def coefficients(self, r):
        """Return ``(tp, ts)`` at pupil radius ``r``, zeroed past the aperture."""
        geom = self.geometry(r)
        return self.coefficients_from_geometry(geom)

    def coefficients_from_geometry(self, geom):
        """Return ``(tp, ts)`` for an already-built :class:`RayGeometry`."""
        o = self.ovoid
        with np.errstate(divide="ignore", invalid="ignore"):
            tp = self._tp(o.n0, o.ni, geom.cos_t0, geom.cos_ti)
            ts = self._ts(o.n0, o.ni, geom.cos_t0, geom.cos_ti)
        zero = np.zeros_like(geom.r)
        return np.where(geom.valid, tp, zero), np.where(geom.valid, ts, zero)

    def reflection_coefficients(self, r):
        """Return ``(rp, rs)`` at pupil radius ``r``."""
        o = self.ovoid
        geom = self.geometry(r)
        with np.errstate(divide="ignore", invalid="ignore"):
            rp = self._rp(o.n0, o.ni, geom.cos_t0, geom.cos_ti)
            rs = self._rs(o.n0, o.ni, geom.cos_t0, geom.cos_ti)
        zero = np.zeros_like(geom.r)
        return np.where(geom.valid, rp, zero), np.where(geom.valid, rs, zero)

    def transmittances(self, r):
        """Return the power transmittances ``(Tp, Ts)`` of Eq. (30)."""
        o = self.ovoid
        geom = self.geometry(r)
        tp, ts = self.coefficients_from_geometry(geom)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.divide(
                o.ni * geom.cos_ti,
                o.n0 * geom.cos_t0,
                out=np.zeros_like(geom.r),
                where=geom.cos_t0 > 0.0,
            )
        return scale * tp**2, scale * ts**2

    def ts(self, r):
        return self.coefficients(r)[1]

    def tp(self, r):
        return self.coefficients(r)[0]
