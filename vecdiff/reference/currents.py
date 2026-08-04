"""Equivalent surface currents on the refracting surface.

The Franz integral of :mod:`vecdiff.reference.stratton_chu` only ever consumes
a pair of equivalent current densities ``(J, M)`` sampled on the surface.
Which physical model produced them is a separate concern, so it lives here
behind a small protocol.

Today the only provider is :class:`PhysicalOpticsCurrents`, which splits the
incident field with the local plane-wave Fresnel coefficients -- the same
hypothesis the stigmatic-refraction note makes.  A rigorous boundary-integral
solve (Mueller/PMCHWT on the body of revolution) would drop that hypothesis;
it would plug in here as another provider without touching the propagator.
"""

from typing import Protocol

import numpy as np

from ..fresnel import FresnelOvoid


class SurfaceCurrents(Protocol):
    """Equivalent currents on Sigma, sampled on an ``(r, phi)`` mesh."""

    def currents(self, geom, phi) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(J, M)`` as ``(n_phi, n_r, 3)`` Cartesian arrays."""
        ...


class PhysicalOpticsCurrents:
    """Local plane-wave Fresnel split of a spherical wave launched from ``A``.

    The incident field is prescribed by its transverse Cartesian components on
    the reference sphere ``G`` -- the same two amplitudes ``vecdiff`` takes as
    input.  Propagation from ``G`` down to the surface is the exact geometrical
    behaviour of a spherical wave from ``A``: amplitude ``|z0| / l0`` and phase
    ``n0 k0 (l0 - |z0|)``.  Nothing about reference spheres downstream, no
    apodization convention, no paraxial step.
    """

    def __init__(self, surface, wavelength, Ex_G, Ey_G=None):
        self.surface = surface
        self.wavelength = float(wavelength)
        self.Ex_G = Ex_G
        self.Ey_G = Ey_G
        self._fresnel = FresnelOvoid(ovoid=surface)

    def incident_field(self, geom, phi):
        """Return the incident field just before the interface, Eq. (36)."""
        surface = self.surface
        k0 = 2.0 * np.pi / self.wavelength

        r = geom.r
        Ex = np.asarray(self.Ex_G(r), dtype=complex)
        Ey = np.zeros_like(Ex) if self.Ey_G is None else np.asarray(self.Ey_G(r), dtype=complex)

        cph = np.cos(phi)
        sph = np.sin(phi)

        # Cartesian -> cylindrical transverse components on G.
        Er = Ex * cph + Ey * sph
        Ephi = -Ex * sph + Ey * cph

        # Eq. (60): the meridional amplitude follows from transversality on G.
        with np.errstate(divide="ignore", invalid="ignore"):
            E_theta0 = np.divide(
                Er,
                geom.cos_a0,
                out=np.zeros_like(Er),
                where=np.abs(geom.cos_a0) > 0.0,
            )

        # Eq. (36): spherical-wave transfer from G down to the interface.
        spread = abs(surface.z0) / geom.l0
        phase = np.exp(1j * surface.n0 * k0 * (geom.l0 - abs(surface.z0)))
        transfer = spread * phase

        return E_theta0 * transfer, Ephi * transfer

    def currents(self, geom, phi):
        surface = self.surface
        E_theta0, Ephi = self.incident_field(geom, phi)

        tp, ts = self._fresnel.coefficients_from_geometry(geom)

        # Eq. (39): the interface acts diagonally in the local s/p frame.
        phi_hat, p0_hat, pi_hat = geom.local_frame(phi)
        E_plus = (tp * E_theta0)[..., None] * pi_hat + (ts * Ephi)[..., None] * phi_hat

        # Locally plane wave along u_i, so H = ni (u_i x E) with Z_vac = 1.
        _, ui_hat = geom.directions(phi)
        H_plus = surface.ni * np.cross(ui_hat, E_plus)

        # Love equivalence with n_V pointing into the image medium.  RayGeometry
        # orients the normal towards the incident medium, so flip it here.
        n_V = -geom.normals(phi)

        J = np.cross(n_V, H_plus)
        M = -np.cross(n_V, E_plus)

        mask = geom.valid[..., None]
        return np.where(mask, J, 0.0), np.where(mask, M, 0.0)


class PrescribedSphericalCapCurrents:
    """Currents from a field prescribed directly on a spherical cap.

    Used to validate the Franz quadrature against a case with a known answer:
    a uniformly apodized converging spherical wave, whose focal field is the
    textbook Airy pattern in the low-numerical-aperture limit.
    """

    def __init__(self, radius, focus_z, n, wavelength, amplitude, polarization=(1.0, 0.0, 0.0)):
        self.radius = float(radius)
        self.focus_z = float(focus_z)
        self.n = float(n)
        self.wavelength = float(wavelength)
        self.amplitude = amplitude
        self.polarization = np.asarray(polarization, dtype=complex)

    def geometry(self, theta, phi):
        """Return positions, inward directions, normals and the area element."""
        R = self.radius
        theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float), np.asarray(phi, dtype=float))
        st, ct = np.sin(theta), np.cos(theta)
        cph, sph = np.cos(phi), np.sin(phi)

        # Cap centred on the focus, opening towards -z (the wave converges to +z).
        Q = np.stack(
            [R * st * cph, R * st * sph, np.broadcast_to(self.focus_z - R * ct, np.shape(st * cph))],
            axis=-1,
        )
        # Propagation direction: from Q towards the focus.
        u = np.stack([-st * cph, -st * sph, np.broadcast_to(ct, np.shape(st * cph))], axis=-1)
        # Outward normal of the cap points away from the focus, i.e. -u.
        n_V = u
        dS = R**2 * st
        return Q, u, n_V, dS

    def currents_on_cap(self, theta, phi):
        Q, u, n_V, dS = self.geometry(theta, phi)

        pol = np.broadcast_to(self.polarization, Q.shape).copy()
        # Project the nominal polarization onto the plane transverse to u.
        pol = pol - np.sum(pol * u, axis=-1)[..., None] * u
        norm = np.sqrt(np.sum(np.abs(pol) ** 2, axis=-1))[..., None]
        pol = np.divide(pol, norm, out=np.zeros_like(pol), where=norm > 0.0)

        E = np.asarray(self.amplitude(theta), dtype=complex)[..., None] * pol
        H = self.n * np.cross(u, E)

        J = np.cross(n_V, H)
        M = -np.cross(n_V, E)
        return Q, J, M, dS
