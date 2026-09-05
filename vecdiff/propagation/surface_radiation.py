"""Homogeneous Maxwell radiation from equivalent surface currents.

This representation propagates supplied boundary data; it does not solve for
that data. The surface normal points into the target region, J=n x H and
M=-n x E. H is normalized by Z0. Direct Green evaluation is valid away from
source nodes; on-surface singular limits require dedicated quadrature.
"""
from dataclasses import dataclass
import numpy as np
from ..fields.electric_spectrum import ElectricSpectrum
from ..fourier.nufft import synthesize


@dataclass(frozen=True)
class SurfaceRadiation:
    sampling: object
    J: np.ndarray
    M: np.ndarray
    wavelength: float
    medium: object

    def __post_init__(self):
        self.medium.wavenumber(self.wavelength)
        for name in ("J", "M"):
            a = np.array(getattr(self, name), complex, copy=True)
            if a.shape != self.sampling.points.shape or not np.isfinite(a).all():
                raise ValueError("surface currents must be finite (nodes,3) arrays")
            a.setflags(write=False); object.__setattr__(self, name, a)

    @classmethod
    def from_boundary(cls, sampling, E, H, wavelength, medium, *, normal_sign=1):
        n = normal_sign*sampling.normals
        return cls(sampling, np.cross(n, H), -np.cross(n, E), wavelength, medium)

    def spectrum(self, grid, *, direction=1, backend="direct", evanescent=False, radial_count=1024, max_order=8):
        """Weyl plane-wave representation outside the source z envelope.

Return a periodic quadrature of the continuous angular spectrum. Grazing
modes are rejected: the Weyl density is singular there. No horizon taper is
applied. Evanescent modes are an explicit quadrature choice for this integral.
        """
        if direction not in (-1, 1): raise ValueError("direction must be +1 or -1")
        k = self.medium.wavenumber(self.wavelength); k0 = 2*np.pi/self.wavelength
        kx, ky = grid.kxy
        gamma2 = k*k-kx*kx-ky*ky
        if np.any(abs(gamma2) < 1e-12*k*k): raise ValueError("spectral quadrature intersects the grazing singularity; change grid")
        mask = np.ones(grid.shape, bool) if evanescent else gamma2 > 0
        gamma = np.sqrt(gamma2[mask].astype(complex))
        wavevectors = np.stack((kx[mask], ky[mask], direction*gamma), axis=-1)
        coeff = np.concatenate((self.J.T, self.M.T))*self.sampling.weights
        # Swapping source and target roles computes integral currents exp(-ik.Q).
        if backend == "polar":
            values = self._polar_transform(wavevectors, k, direction, radial_count, max_order)
        elif np.any(wavevectors.imag):
            if backend != "direct": raise ValueError("evanescent radiation requires direct quadrature")
            values = np.empty((6, len(wavevectors)), complex)
            for start in range(0, len(wavevectors), 512):
                values[:, start:start+512] = coeff @ np.exp(-1j*self.sampling.points @ wavevectors[start:start+512].T)
        else:
            values = synthesize(-self.sampling.points, coeff, wavevectors.real, backend=backend)
        j, m = values[:3].T, values[3:].T
        transverse_j = j-wavevectors*np.sum(wavevectors*j, axis=-1)[:, None]/k**2
        a = (-k0*transverse_j+np.cross(wavevectors, m))/(2*gamma[:, None]*grid.period_area)
        return ElectricSpectrum(wavevectors, a, self.wavelength, self.medium)

    def _polar_transform(self, wavevectors, k, direction, radial_count, max_order):
        from scipy.interpolate import CubicSpline
        from ..fourier.polar import cylindrical_transform
        shape = self.sampling.parameter_shape
        if shape is None or len(shape) != 2: raise ValueError("polar backend requires tensor-product surface sampling")
        frame = self.sampling.surface.frame
        if not np.allclose(frame.rotation, np.eye(3), atol=1e-14):
            raise ValueError("polar backend currently requires the surface axis parallel to global z")
        local = (self.sampling.points-frame.origin).reshape(shape+(3,))
        radius = np.linalg.norm(local[:, 0, :2], axis=-1)
        phi = np.arange(shape[1])*2*np.pi/shape[1]
        expected = np.stack((radius[:, None]*np.cos(phi), radius[:, None]*np.sin(phi),
                             np.broadcast_to(local[:, :1, 2], shape)), axis=-1)
        if not np.allclose(local, expected, atol=1e-10, rtol=1e-10):
            raise ValueError("polar backend requires full 0..2pi rings at each radial node")
        if np.any(wavevectors.imag): raise ValueError("polar evanescent quadrature is not implemented; use direct")
        kr = np.linalg.norm(wavevectors[:, :2].real, axis=-1)
        # Angle parameterization is smooth at BOTH kr=0 and kz=0.
        theta = np.arctan2(kr, abs(wavevectors[:, 2]))
        theta_tab = np.linspace(0, np.max(theta), radial_count)
        kz_tab, kr_tab = k*np.cos(theta_tab), k*np.sin(theta_tab)
        c = (np.concatenate((self.J.T, self.M.T))*self.sampling.weights).reshape((6,)+shape)
        orders, coeff = cylindrical_transform(radius, local[:, 0, 2], c, kr_tab, direction*kz_tab,
                                               max_order=max_order)
        angle = np.arctan2(wavevectors[:, 1].real, wavevectors[:, 0].real)
        out = np.zeros((6, len(kr)), complex)
        for m, channel in zip(orders, coeff):
            out += CubicSpline(theta_tab, channel, axis=-1)(theta)*np.exp(1j*m*angle)
        return out*np.exp(-1j*wavevectors @ frame.origin)[None]

    def angular_spectrum(self, *, direction=1, n_theta=100, n_phi=160, backend="direct", radial_count=1024, max_order=8):
        """Continuous propagating hemisphere quadrature, without FFT periodicity.

Gauss-Legendre theta and periodic phi integrate d^2k/(2*pi)^2. Its Jacobian
cancels the Weyl 1/kz singularity. This explicitly excludes evanescent waves;
use spectrum(..., evanescent=True) when those are required. Angular quadrature
must resolve both the source extent and the requested observation extent.
        """
        from numpy.polynomial.legendre import leggauss
        if direction not in (-1, 1): raise ValueError("direction must be +1 or -1")
        if n_theta < 2 or n_phi < 2: raise ValueError("angular sampling requires at least two nodes per axis")
        t, wt = leggauss(n_theta); t = (t+1)*np.pi/4; wt *= np.pi/4
        t, p = np.meshgrid(t, np.arange(n_phi)*2*np.pi/n_phi, indexing="ij")
        k = self.medium.wavenumber(self.wavelength); k0 = 2*np.pi/self.wavelength
        kv = k*np.stack((np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), direction*np.cos(t)), axis=-1).reshape(-1, 3)
        if backend == "polar":
            values = self._polar_transform(kv.astype(complex), k, direction, radial_count, max_order)
        else:
            c = np.concatenate((self.J.T, self.M.T))*self.sampling.weights
            values = synthesize(-self.sampling.points, c, kv, backend=backend)
        j, m = values[:3].T, values[3:].T
        tj = j-kv*np.sum(kv*j, axis=-1)[:, None]/k**2
        weights = (k*np.sin(t)*wt[:, None]*(2*np.pi/n_phi)/(2*(2*np.pi)**2)).ravel()
        a = (-k0*tj+np.cross(kv, m))*weights[:, None]
        return ElectricSpectrum(kv, a, self.wavelength, self.medium)

    def evaluate_propagating(self, points, *, direction=1, n_theta=100,
                             radial_count=1024, max_order=8):
        """Propagating hemisphere via analytic azimuthal Fourier–Bessel synthesis.

For full axisymmetric rings with a resolved current harmonic tail. Equivalent
to angular_spectrum(..., backend='polar').evaluate at converged azimuthal
quadrature, without allocating O(n_theta*n_phi_observation) modes. The source
translation is removed before harmonic analysis and restored in observations.
This excludes evanescent fields; evaluation within a source z envelope is only
an analytic continuation diagnostic, not a physical boundary limit.
        """
        from ..fourier.polar import cylindrical_synthesize
        p = np.asarray(points, float)
        if p.shape[-1:] != (3,) or not np.isfinite(p).all():
            raise ValueError("points must be finite (...,3)")
        if not isinstance(max_order, int) or max_order < 0:
            raise ValueError("max_order must be a nonnegative integer")
        # The dyadic projection adds at most two harmonics; H=k x E/k0
        # adds one more. The source-current tail itself is checked upstream.
        n_phi = 2*(max_order+3)+1
        spectrum = self.angular_spectrum(direction=direction, n_theta=n_theta,
                    n_phi=n_phi, backend="polar", radial_count=radial_count, max_order=max_order)
        origin = self.sampling.surface.frame.origin
        values = np.concatenate((spectrum.amplitudes, spectrum.magnetic_amplitudes), axis=-1)
        values *= np.exp(1j*spectrum.wavevectors @ origin)[:, None]
        # Amplitudes already contain Delta-phi. fft(values) gives Fourier
        # coefficients multiplied by the full azimuthal measure 2*pi.
        coefficients = np.fft.fft(values.reshape(n_theta, n_phi, 6), axis=1)
        orders = np.rint(np.fft.fftfreq(n_phi)*n_phi).astype(int)
        k = spectrum.wavevectors.real.reshape(n_theta, n_phi, 3)[:, 0]
        out = cylindrical_synthesize(np.linalg.norm(k[:, :2], axis=-1), k[:, 2],
                                    coefficients, orders, p-origin)
        return out[..., :3], out[..., 3:]

    def evaluate(self, points, *, chunk=16):
        """Direct dyadic Green integral, including reactive near-field terms.

Cost is O(Npoints*Nsurface). Users must converge quadrature and their distance
from the surface independently; this is not a singular boundary evaluator.
        """
        p = np.asarray(points, float); shape = p.shape; p = p.reshape(-1, 3)
        k = self.medium.wavenumber(self.wavelength); k0 = 2*np.pi/self.wavelength
        jw, mw = self.J*self.sampling.weights[:, None], self.M*self.sampling.weights[:, None]
        out_e, out_h = np.empty_like(p, complex), np.empty_like(p, complex)
        for start in range(0, len(p), chunk):
            rv = p[start:start+chunk, None]-self.sampling.points[None]
            r = np.linalg.norm(rv, axis=-1)
            if np.any(r == 0): raise ValueError("on-node Green evaluation is singular")
            u = rv/r[..., None]
            g = np.exp(1j*k*r)/(4*np.pi*r)
            gp = (1j*k-1/r)*g
            gpp = ((1j*k-1/r)**2+1/r**2)*g
            def dyadic(c):
                dot = np.sum(u*c, axis=-1)
                hess = gpp[..., None]*dot[..., None]*u+(gp/r)[..., None]*(c-dot[..., None]*u)
                return g[..., None]*c+hess/k**2
            grad = gp[..., None]*u
            out_e[start:start+chunk] = np.sum(1j*k0*dyadic(jw)-np.cross(grad, mw), axis=1)
            out_h[start:start+chunk] = np.sum(np.cross(grad, jw)+1j*k0*self.medium.epsilon_r*dyadic(mw), axis=1)
        return out_e.reshape(shape), out_h.reshape(shape)
