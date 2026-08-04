# CartesianSurfaces.py

import numpy as np

from .GOTS_parameters import GOTS_params


class CartesianSurface:
    """The stigmatic Cartesian oval separating media ``n0`` and ``ni``.

    Parametrisation
    ---------------
    The closed form ``z(rho)`` is written in terms of ``rho``, the *spherical*
    radius measured from the vertex ``O``, so that ``rho**2 = r**2 + z**2``
    where ``r`` is the cylindrical radius.  This is verifiable: the stigmatic
    condition ``n0*l0 + ni*li = n0*|z0| + ni*|zi|`` holds to machine zero under
    that reading and fails badly under any other.

    Callers, however, work with ``r`` -- the physical aperture coordinate, the
    radius of the stop sitting on the diopter.  :meth:`rho_from_r` bridges the
    two.  Inverting ``r(rho)`` in closed form means solving a quartic, so it is
    done numerically instead: a dense monotone table followed by Newton
    refinement, which reaches machine precision.
    """

    #: Samples used to tabulate the monotone branch of ``r(rho)``.
    _TABLE_SIZE = 200_001

    def __init__(self, n0, ni, z0, zi):
        self.n0 = n0
        self.ni = ni
        self.z0 = z0
        self.zi = zi

        (
        self.G,
        self.O,
        self.T,
        self.S
        ) = GOTS_params(self.n0, self.z0, self.ni, self.zi)

        self._inverse_table = None

    def z(self, rho):
        rho = np.asarray(rho, dtype=float)

        G = self.G
        O = self.O
        T = self.T
        S = self.S

        return ((O + T*rho**2)*rho**2) / (
            1
            + S*rho**2
            + np.sqrt(1 + (2*S - O**2*G)*rho**2)
        )

    def dz(self, rho):
        rho = np.asarray(rho, dtype=float)

        G = self.G
        O = self.O
        T = self.T
        S = self.S

        Q = np.sqrt(1 + (2*S - O**2*G)*rho**2)

        N = (O + T*rho**2)*rho**2
        D = 1 + S*rho**2 + Q

        Np = 2*O*rho + 4*T*rho**3
        Qp = (2*S - O**2*G)*rho / Q
        Dp = 2*S*rho + Qp

        return (Np*D - N*Dp) / D**2

    def r(self, rho):
        rho = np.asarray(rho, dtype=float)
        z = self.z(rho)
        return np.sqrt(rho**2 - z**2)

    # ------------------------------------------------------------------ #
    #  Inversion r -> rho                                                  #
    # ------------------------------------------------------------------ #

    def _dr_drho(self, rho):
        """Return ``dr/drho`` from ``r**2 = rho**2 - z**2``."""
        rho = np.asarray(rho, dtype=float)
        z = self.z(rho)
        r = np.sqrt(np.maximum(rho**2 - z**2, 0.0))
        return np.divide(
            rho - z * self.dz(rho),
            r,
            out=np.ones_like(rho),
            where=r > 0.0,
        )

    def _build_inverse_table(self):
        """Tabulate the monotone increasing branch ``rho -> r``."""
        # The closed form stops being real once the inner square root or
        # ``rho**2 - z**2`` turns negative; find that ceiling first.
        span = 4.0 * max(abs(self.z0), abs(self.zi))
        probe = np.linspace(0.0, span, self._TABLE_SIZE)
        with np.errstate(invalid="ignore", divide="ignore"):
            r_probe = self.r(probe)
        finite = np.isfinite(r_probe)
        if not finite[0]:
            raise ValueError("surface is degenerate at the vertex.")
        end = int(np.argmin(finite)) if not np.all(finite) else probe.size
        probe = probe[:end]
        r_probe = r_probe[:end]

        # Keep the branch up to the fold dr/drho = 0, where r stops increasing.
        rising = np.flatnonzero(np.diff(r_probe) <= 0.0)
        fold = int(rising[0]) + 1 if rising.size else probe.size - 1

        rho_table = probe[: fold + 1]
        r_table = r_probe[: fold + 1]

        # np.interp needs a strictly increasing abscissa.
        keep = np.concatenate(([True], np.diff(r_table) > 0.0))
        self._inverse_table = (r_table[keep], rho_table[keep])
        return self._inverse_table

    @property
    def _table(self):
        if self._inverse_table is None:
            self._build_inverse_table()
        return self._inverse_table

    @property
    def rho_fold(self) -> float:
        """Spherical radius at which ``r(rho)`` stops increasing."""
        return float(self._table[1][-1])

    @property
    def r_fold(self) -> float:
        """Largest cylindrical radius the surface reaches, ``r(rho_fold)``."""
        return float(self._table[0][-1])

    def rho_from_r(self, r, *, return_valid: bool = False, newton_steps: int = 3):
        """Return the surface parameter ``rho`` for cylindrical radius ``r``.

        Inversion is numerical -- the analytic route is a quartic -- but exact
        to machine precision: interpolation on a dense monotone table seeded
        into Newton iterations on ``r(rho) - r``.

        Samples with ``r`` beyond :attr:`r_fold` have no preimage; they are
        clamped to the fold and reported through ``return_valid``.
        """
        r = np.asarray(r, dtype=float)
        r_table, rho_table = self._table

        valid = (r >= 0.0) & (r <= r_table[-1])
        r_clamped = np.clip(r, 0.0, r_table[-1])

        rho = np.interp(r_clamped, r_table, rho_table)

        # Newton polish.  Near the fold dr/drho vanishes, so guard the step.
        for _ in range(int(newton_steps)):
            with np.errstate(invalid="ignore", divide="ignore"):
                residual = self.r(rho) - r_clamped
                slope = self._dr_drho(rho)
            step = np.divide(
                residual,
                slope,
                out=np.zeros_like(rho),
                where=np.abs(slope) > 1e-9,
            )
            rho = np.clip(rho - step, 0.0, rho_table[-1])

        rho = np.where(r_clamped > 0.0, rho, 0.0)

        if return_valid:
            return rho, valid
        return rho

    # ------------------------------------------------------------------ #
    #  Convenience geometry                                                #
    # ------------------------------------------------------------------ #

    def sag(self, r):
        """Return the axial sag ``z`` at cylindrical radius ``r``."""
        return self.z(self.rho_from_r(r))

    def ray_geometry(self, r):
        """Return the :class:`~vecdiff.geometry.RayGeometry` on the grid ``r``."""
        from .geometry import ray_geometry

        return ray_geometry(self, r)

    @property
    def aperture_limit(self) -> float:
        """Largest usable pupil radius, set by grazing incidence."""
        cached = getattr(self, "_aperture_limit", None)
        if cached is None:
            from .geometry import grazing_radius

            cached = grazing_radius(self)
            self._aperture_limit = cached
        return cached
