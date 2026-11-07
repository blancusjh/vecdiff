"""Beams: exact source-free fields built from a scalar profile and a polarization.

A beam is a superposition of propagating plane waves in one homogeneous region,

    E(r) = integral over directions s of F(s) exp(i k s . (r - origin)) dOmega,

placed by a Frame whose normal is the propagation axis. The kinds differ only
in where the field is prescribed. Profile and polarization stay separate: the
prescribed transverse field is profile(x, y) times the polarization state at
the point's azimuth. Quadratures are chosen from the region to be evaluated.
"""

import numpy as np

from vecdiff_general._arrays import positive
from vecdiff_general._backend import bd, jv, leggauss, to_numpy
from vecdiff_general.fields.carrier import Carrier
from vecdiff_general.fields.polarization import Jones
from vecdiff_general.fields.spectrum import PlaneWaveExpansion
from vecdiff_general.geometry.frames import Frame
from vecdiff_general.media.medium import HomogeneousRegion

# Relative size below which a density's azimuthal harmonic counts as absent.
_ROUNDOFF = 64 * np.finfo(float).eps


class _Beam:
    def __init__(self, profile, polarization, frame, wavelength0, region):
        if not callable(profile):
            raise TypeError("profile must be a callable p(x, y) from fields.profiles")

        if not isinstance(polarization, Jones):
            raise TypeError("polarization must be a state from fields.polarization")

        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame")

        if not isinstance(region, HomogeneousRegion):
            raise TypeError("a beam lives in one HomogeneousRegion")

        self.profile, self.polarization, self.frame = profile, polarization, frame
        self.wavelength0 = positive(wavelength0, "vacuum wavelength")
        self.region = region
        self.wavenumber = 2 * bd.pi * region.material.n / self.wavelength0

    def _transverse(self, x, y):
        """Prescribed (u, v) field components at frame coordinates x, y."""
        amplitude = bd.asarray(self.profile(x, y), dtype=complex)

        return amplitude[..., None] * self.polarization.vectors(bd.arctan2(y, x))

    def _directions(self, theta, phi):
        """Unit direction and its polar and azimuthal unit vectors in global axes."""
        st, ct = bd.sin(theta)[:, None], bd.cos(theta)[:, None]
        sp, cp = bd.sin(phi)[None], bd.cos(phi)[None]
        u, v, n = self.frame.u, self.frame.v, self.frame.normal

        def vector(a, b, c):
            return a[..., None] * u + b[..., None] * v + c[..., None] * n

        zero = bd.zeros(bd.broadcast_shapes(st.shape, sp.shape))

        return (
            vector(st * cp, st * sp, ct + zero),
            vector(ct * cp, ct * sp, -st + zero),
            vector(-sp + zero, cp + zero, zero),
        )

    def _azimuthal_order(self, theta, extent):
        """Highest azimuthal harmonic of the density present above roundoff."""
        count = 32

        while True:
            phi = 2 * bd.pi * bd.arange(count) / count
            harmonics = bd.abs(bd.fft.fft(self._density(theta, phi, extent), axis=1))
            present = bd.max(harmonics, axis=(0, 2)) > _ROUNDOFF * harmonics.max()
            m = bd.fft.fftfreq(count, 1 / count)
            order = int(bd.abs(m[present]).max()) if present.any() else 0

            if order < count // 4 or count >= 4096:
                return order

            count *= 2

    def _polar_rule(self, radius):
        """Polar nodes and dtheta weights resolving k s.(r - origin) for |r - origin| <= radius."""
        radius = float(radius)

        if not bd.isfinite(radius) or radius < 0:
            raise ValueError("quadrature radius must be finite and nonnegative")

        extent = self._extent()
        top = self._theta_max()
        excursion = self.wavenumber * (radius * top + extent * bd.sin(top))
        count = int(bd.ceil(excursion / 2 + 8 * excursion ** (1 / 3) + 24))
        t, w = leggauss(count)

        return top / 2 * (t + 1), top / 2 * w, extent, top

    def _harmonics(self, radius):
        """Polar nodes, sin(theta) dtheta weights and the density's azimuthal harmonics.

        F(theta, phi) = sum_m exp(i m phi) (F_theta[m] theta_hat + F_phi[m] phi_hat)
        on the frame axes, as {m: (F_theta, F_phi)} for the orders present
        above roundoff; exact for |r - origin| <= radius like quadrature(radius).
        The polar rule is composite, 64 nodes per panel with at most 120
        radians of phase each, so large radii cost no large Gauss rule.
        theta_hat and phi_hat turn once per revolution, so their orders reach
        one beyond the Cartesian components' highest order.
        """
        radius = float(radius)

        if not bd.isfinite(radius) or radius < 0:
            raise ValueError("quadrature radius must be finite and nonnegative")

        extent, top = self._extent(), self._theta_max()
        excursion = self.wavenumber * (radius * top + extent * bd.sin(top))
        panels = max(1, int(np.ceil(float(excursion) / 120)))
        t, w = leggauss(64)
        edges = top * bd.arange(panels + 1) / panels
        half = (edges[1:] - edges[:-1])[:, None] / 2
        theta = ((edges[:-1, None] + half) + half * t[None]).ravel()
        w = (half * w[None]).ravel()
        order = self._azimuthal_order(theta[:: max(1, len(theta) // 16)], extent) + 1
        count = 2 * order + 2
        phi = 2 * bd.pi * bd.arange(count) / count
        density = self._density(theta, phi, extent)
        _, theta_hat, phi_hat = self._directions(theta, phi)
        parts = [
            bd.fft.fft(bd.sum(density * axis, -1), axis=1) / count
            for axis in (theta_hat, phi_hat)
        ]
        size = to_numpy(
            bd.maximum(bd.max(bd.abs(parts[0]), 0), bd.max(bd.abs(parts[1]), 0))
        )
        m = np.rint(np.fft.fftfreq(count, 1 / count)).astype(int)
        present = (size > _ROUNDOFF * size.max()) & (abs(m) <= order)

        return (
            theta,
            w * bd.sin(theta),
            {
                int(m[j]): (parts[0][:, j], parts[1][:, j])
                for j in np.flatnonzero(present)
            },
        )

    def quadrature(self, radius):
        """Direction nodes theta, phi, weights of dOmega and density values F.

        Exact to roundoff for |r - origin| <= radius: the polar rule resolves
        the phase k s.(r - origin) and the density structure, the periodic
        azimuthal rule their combined harmonics.
        """
        theta, w, extent, top = self._polar_rule(radius)
        count = len(theta)
        k = self.wavenumber
        radius = float(radius)
        reach = k * radius * bd.sin(top)
        orders = self._azimuthal_order(theta[:: max(1, count // 16)], extent)
        azimuths = int(orders + reach + 10 * reach ** (1 / 3) + 16)
        azimuths += azimuths % 2
        phi = 2 * bd.pi * bd.arange(azimuths) / azimuths
        weight = (w * bd.sin(theta))[:, None] * bd.full(azimuths, 2 * bd.pi / azimuths)

        return theta, phi, weight, self._density(theta, phi, extent)

    def plane_waves(self, radius):
        """PlaneWaveExpansion equal to the beam wherever |r - origin| <= radius."""
        theta, phi, weight, density = self.quadrature(radius)
        s, _, _ = self._directions(theta, phi)
        keep = bd.any(density != 0, axis=-1)

        return PlaneWaveExpansion(
            self.wavenumber * s[keep],
            (weight[..., None] * density)[keep],
            self.frame.origin,
            self.wavelength0,
            self.region,
        )

    def evaluate(self, grid, **kwargs):
        """SI E and H on any grid; keyword arguments select the Fourier summation method."""
        reach = bd.max(bd.linalg.norm(grid.points - self.frame.origin, axis=-1))

        return self.plane_waves(reach).evaluate(grid, **kwargs)


class PlaneBeam(_Beam):
    """Beam whose transverse electric field is prescribed on the plane of its frame.

    On that plane E_u, E_v equal profile(x, y) times the polarization state;
    E_z follows from transversality of each plane wave. Only the propagating
    part of the prescribed field is kept, so a structure finer than a
    wavelength is low-passed. ``support`` is the radius beyond which the
    profile vanishes, needed for callables that do not declare one. An
    unbounded Uniform profile with Cartesian polarization is one plane wave.
    """

    def __init__(
        self,
        profile,
        polarization,
        *,
        wavelength0,
        region,
        frame=None,
        support=None,
    ):
        super().__init__(profile, polarization, frame or Frame(), wavelength0, region)
        support = getattr(profile, "support", None) if support is None else support

        if support is not None:
            support = positive(support, "profile support")
        elif not self.is_plane_wave:
            raise ValueError(
                "this profile declares no support; pass support= (metres), "
                "or use an unbounded Uniform profile with Cartesian polarization"
            )

        self.support = support

    @property
    def carrier(self):
        return Carrier.collimated(self.frame.normal, self.frame.origin)

    @property
    def is_plane_wave(self):
        """An unbounded uniform profile with Cartesian polarization: one plane wave."""
        return (
            getattr(self.profile, "support", 0) is None
            and getattr(self.profile, "bandwidth", 0) is None
            and self.polarization.basis == "cartesian"
        )

    def _extent(self):
        if self.support is None:
            raise ValueError(
                "a plane wave has no direction quadrature; use plane_waves"
            )

        return self.support

    def _theta_max(self):
        bandwidth = getattr(self.profile, "bandwidth", None)

        if bandwidth is None or bandwidth >= self.wavenumber:
            return bd.pi / 2

        return float(bd.arcsin(bandwidth / self.wavenumber))

    def plane_waves(self, radius):
        if self.support is not None:
            return super().plane_waves(radius)

        a, b = self.polarization.components
        amplitude = complex(bd.asarray(self.profile(0.0, 0.0)))

        return PlaneWaveExpansion(
            [self.wavenumber * self.frame.normal],
            [amplitude * (a * self.frame.u + b * self.frame.v)],
            self.frame.origin,
            self.wavelength0,
            self.region,
        )

    def _density(self, theta, phi, extent):
        """F = k² cos(theta) E~(k_t) / (2 pi)² from the 2D Fourier transform.

        The prescribed field is split into azimuthal harmonics c_m(rho) on a
        polar quadrature; each transforms by a Hankel integral of order m:
        FT[c_m(rho) e^{i m phi}] = 2 pi (-i)^m e^{i m phi_k} int c_m J_m(k_t rho) rho drho.
        """
        k = self.wavenumber
        reach = k * extent * bd.sin(bd.max(theta))
        count = int(bd.ceil(reach / 2 + 8 * reach ** (1 / 3) + 24))
        t, w = leggauss(count)
        rho, w = extent / 2 * (t + 1), extent / 2 * w * extent / 2 * (t + 1)
        harmonics = self._plane_harmonics(rho)
        m = bd.fft.fftfreq(harmonics.shape[1], 1 / harmonics.shape[1]).astype(int)
        present = bd.max(bd.abs(harmonics), axis=(0, 2)) > _ROUNDOFF * bd.max(
            bd.abs(harmonics)
        )
        kt = k * bd.sin(theta)
        spectrum = bd.zeros((len(theta), len(phi), 2), complex)

        for order in m[present]:
            hankel = (jv(order, kt[:, None] * rho[None]) * w) @ harmonics[:, order]
            spectrum += (
                2
                * bd.pi
                * (-1j) ** order
                * bd.exp(1j * order * phi)[None, :, None]
                * hankel[:, None]
            )

        s, _, _ = self._directions(theta, phi)
        st, ct = bd.sin(theta)[:, None], bd.cos(theta)[:, None]
        cp, sp = bd.cos(phi)[None], bd.sin(phi)[None]
        normal = -st * (cp * spectrum[..., 0] + sp * spectrum[..., 1])

        return (
            k**2
            / (2 * bd.pi) ** 2
            * (
                (ct * spectrum[..., 0])[..., None] * self.frame.u
                + (ct * spectrum[..., 1])[..., None] * self.frame.v
                + normal[..., None] * self.frame.normal
            )
        )

    def _plane_harmonics(self, rho):
        """Azimuthal Fourier coefficients c_m(rho) of (E_u, E_v), shape (rho, m, 2)."""
        count = 32

        while True:
            phi = 2 * bd.pi * bd.arange(count) / count
            x = rho[:, None] * bd.cos(phi)[None]
            y = rho[:, None] * bd.sin(phi)[None]
            harmonics = bd.fft.fft(self._transverse(x, y), axis=1) / count
            magnitude = bd.max(bd.abs(harmonics), axis=(0, 2))
            tail = magnitude[count // 4 : 3 * count // 4]

            if bd.all(tail <= _ROUNDOFF * magnitude.max()) or count >= 4096:
                return harmonics

            count *= 2


class FocusedBeam(_Beam):
    """Field focused by an ideal lens whose entrance pupil carries the prescribed field.

    The frame origin is the focus and its normal the optical axis. The ray of
    direction s leaves the pupil point -focal_length sin(theta) (cos phi,
    sin phi) (sine condition); its pupil field profile(x, y) times the
    polarization is carried to s by the minimal rotation from the axis to s,
    so radial stays meridional and azimuthal stays azimuthal. Directions up to
    sin(theta) = max_sine are admitted (Debye integral):

        E(r) = -(i k f / 2 pi) integral A(theta) R(s) E_pupil exp(i k s.(r - focus)) dOmega.

    apodization "aplanatic" is A = sqrt(cos theta): power through the pupil
    equals power through the focal region. "none" is A = cos theta, so the
    transverse-wavevector spectrum is the pupil field itself (the manuscript's
    Appendix C convention). On the reference sphere |r - focus| = f the field
    is A(theta) times the pupil field.
    """

    def __init__(
        self,
        profile,
        polarization,
        *,
        focal_length,
        max_sine,
        wavelength0,
        region,
        frame=None,
        apodization="aplanatic",
    ):
        super().__init__(profile, polarization, frame or Frame(), wavelength0, region)
        self.focal_length = positive(focal_length, "focal length")
        max_sine = float(max_sine)

        if not 0 < max_sine <= 1:
            raise ValueError("max_sine must lie in (0, 1]")

        if apodization not in ("aplanatic", "none"):
            raise ValueError("apodization must be 'aplanatic' or 'none'")

        self.max_sine, self.apodization = max_sine, apodization

    @property
    def focus(self):
        return self.frame.origin

    @property
    def carrier(self):
        return Carrier.converging(self.focus)

    def _extent(self):
        return 0.0

    def _theta_max(self):
        sine = self.max_sine
        support = getattr(self.profile, "support", None)

        if support is not None:
            sine = min(sine, support / self.focal_length)

        return float(bd.arcsin(sine))

    def _density(self, theta, phi, extent):
        f, k = self.focal_length, self.wavenumber
        s, theta_hat, phi_hat = self._directions(theta, phi)
        radius = f * bd.sin(theta)[:, None]
        pupil = self._transverse(
            -radius * bd.cos(phi)[None], -radius * bd.sin(phi)[None]
        )
        cp, sp = bd.cos(phi)[None], bd.sin(phi)[None]
        meridional = pupil[..., 0] * cp + pupil[..., 1] * sp
        azimuthal = -pupil[..., 0] * sp + pupil[..., 1] * cp
        cosine = bd.cos(theta)[:, None]
        apodization = bd.sqrt(cosine) if self.apodization == "aplanatic" else cosine
        scale = -1j * k * f / (2 * bd.pi) * apodization

        return (scale * meridional)[..., None] * theta_hat + (scale * azimuthal)[
            ..., None
        ] * phi_hat
