"""Homogeneous propagation and completion of plane-sampled electric fields."""
import numpy as np
from ..fields.electric_spectrum import ElectricSpectrum
from ..fourier.cartesian import transform, inverse
from ..fields.electric_field import ElectricField
from ..geometry.domains import PlaneDomain
from ..geometry.frames import Frame
from ..sampling.grids import CartesianGrid


def _spectrum_at_plane(field, *, direction=1):
    if direction not in (-1, 1): raise ValueError("direction must be +1 or -1")
    if not isinstance(field.domain, PlaneDomain) or not isinstance(field.sampling, CartesianGrid):
        raise TypeError("FFT spectrum requires a PlaneDomain and CartesianGrid")
    grid, frame = field.sampling, field.domain.frame
    kx, ky = grid.kxy
    kn = field.medium.wavenumber(field.wavelength)
    kz = direction*np.sqrt((kn**2-kx*kx-ky*ky).astype(complex))
    klocal = np.stack((kx, ky, kz), axis=-1).reshape(-1, 3)
    k = frame.vectors(klocal)
    ax, ay = (transform(v, grid).ravel()/grid.period_area for v in (field.Ex, field.Ey))
    if field.Ez is None:
        grazing = abs(k[:, 2]) < 1e-12*kn
        # Even when kx Ex+ky Ey=0, Ez is not determined on this set.
        if np.any(grazing & ((abs(ax)+abs(ay)) > 1e-13*max(1, np.max(abs(ax)+abs(ay))))):
            raise ValueError("Ez completion is not unique for a populated mode with global kz=0")
        az = np.divide(-k[:, 0]*ax-k[:, 1]*ay, k[:, 2], out=np.zeros_like(ax), where=~grazing)
    else:
        az = transform(field.Ez, grid).ravel()/grid.period_area
    a = np.stack((ax, ay, az), axis=-1)
    # Remove exact zeros only; no hidden spectral threshold or horizon taper.
    used = np.any(a != 0, axis=-1)
    return ElectricSpectrum(k[used], a[used], field.wavelength, field.medium)


def spectrum_of(field, *, direction=1):
    """Spectrum in global coordinates, with amplitudes referenced to global zero.

For appreciable evanescent content far from global zero, this representation
can be ill-conditioned. Same-grid propagate() instead uses a local reference
plane and does not perform exponentially growing coordinate re-referencing.
    """
    spec = _spectrum_at_plane(field, direction=direction)
    return spec.translated(-field.domain.frame.origin)


def propagate(field, distance, *, direction=1, backend="direct"):
    """Evaluate the same spectrum on a plane displaced along its local normal.

Evanescent waves are retained. Propagation against their decay direction can
be exponentially ill-conditioned; this routine never silently filters them.
    """
    spec = _spectrum_at_plane(field, direction=direction)
    frame = field.domain.frame
    domain = PlaneDomain(Frame(frame.origin + distance*frame.rotation[:, 2], frame.rotation))
    displacement = distance*frame.rotation[:, 2]
    if backend != "direct":
        relative_domain = PlaneDomain(Frame(displacement, frame.rotation))
        e, _ = spec.evaluate(relative_domain.points(field.sampling), backend=backend)
        return ElectricField(e[..., 0], e[..., 1], field.sampling, domain,
                             field.wavelength, field.medium, e[..., 2])
    # Homogeneous propagation on the original lattice is an FFT operation.
    grid = field.sampling
    local_k = spec.wavevectors @ frame.rotation
    ix = np.rint(local_k[:, 0].real*grid.dx*len(grid.x)/(2*np.pi)).astype(int) % len(grid.x)
    iy = np.rint(local_k[:, 1].real*grid.dy*len(grid.y)/(2*np.pi)).astype(int) % len(grid.y)
    coefficients = np.zeros((3,)+grid.shape, complex)
    coefficients[:, iy, ix] = (spec.amplitudes*np.exp(1j*spec.wavevectors @ displacement)[:, None]).T*grid.period_area
    e = inverse(coefficients, grid)
    return ElectricField(e[0], e[1], grid, domain, field.wavelength, field.medium, e[2])
