"""Ideal aplanatic imaging between conjugate planes in homogeneous regions.

This is an isoplanatic pupil model, not a lens-surface or mask-scattering solver.
The pupil acts in the transported (meridional p, azimuthal s) basis. Its Jacobian
preserves normal power when the pupil is unitary and all input modes are passed.
"""

import numpy as np

from .._arrays import positive
from ..fields.spectrum import PlaneSpectrum, PlaneWaveExpansion
from ..fourier.cartesian import FFT2, IFFT2, frequency_grid
from ..geometry import Frame
from ..media import HomogeneousRegion
from ..sampling import CartesianGrid


def _pupil_geometry(grid, wavelength0, object_index, image_index, magnification, na):
    wavelength0 = positive(wavelength0, "wavelength0")
    na = positive(na, "numerical_aperture")
    magnification = float(magnification)

    if not np.isfinite(magnification) or magnification == 0:
        raise ValueError("magnification must be finite and nonzero")

    if na >= image_index or abs(magnification) * na >= object_index:
        raise ValueError("both object and image apertures must stay below grazing")

    fx, fy = frequency_grid(grid.coordinates)
    qu = wavelength0 * fx
    qv = wavelength0 * fy
    radial = np.hypot(qu, qv)
    pupil_radius = abs(magnification) * na
    du, dv = grid.spacing

    # The pupil must fit strictly inside the sampled spectral rectangle.
    if pupil_radius >= min(wavelength0 / (2 * du), wavelength0 / (2 * dv)):
        raise ValueError("grid does not resolve the requested object-side pupil")

    support = radial <= pupil_radius
    cos_object = np.sqrt(np.maximum(0, 1 - (radial / object_index) ** 2))
    cos_image = np.sqrt(
        np.maximum(0, 1 - (radial / (abs(magnification) * image_index)) ** 2)
    )
    gain = np.zeros(grid.shape)
    gain[support] = np.sqrt(
        object_index * cos_object[support] / (image_index * cos_image[support])
    ) / abs(magnification)

    return qu, qv, cos_object, cos_image, support, gain


def _invert_frequencies(values, magnification):
    if magnification > 0:
        return values

    ny, nx = values.shape[:2]
    rows = (2 * (ny // 2) - np.arange(ny)) % ny
    columns = (2 * (nx // 2) - np.arange(nx)) % nx

    return values[rows][:, columns]


def image_spectrum(
    source,
    image_region,
    *,
    magnification,
    numerical_aperture,
    image_frame=None,
    pupil=None,
):
    """Map a forward PlaneSpectrum to an ideal conjugate image PlaneSpectrum.

    Signed transverse magnification maps x_image = magnification * x_object.
    numerical_aperture is image-side n sin(theta). image_frame locates the image
    plane; omitted means the same reference frame, not a predicted image distance.
    Grid spacing scales by abs(magnification); the pupil limits spatial bandwidth.

    pupil is a scalar complex amplitude or grid.shape + (2, 2) Jones operator in
    local (p, s) mode bases, sampled on the OBJECT frequency grid. Nonunitary
    pupils change throughput. The caller specifies a physically achievable pupil.
    Evanescent and out-of-pupil input modes do not enter this imaging model.
    """
    if not isinstance(source, PlaneSpectrum):
        raise TypeError("source must be a PlaneSpectrum")

    if not isinstance(image_region, HomogeneousRegion):
        raise TypeError("image_region must be a HomogeneousRegion")

    grid = source.grid
    expansion = source.expansion
    image_frame = grid.frame if image_frame is None else image_frame

    if not isinstance(image_frame, Frame):
        raise TypeError("image_frame must be a Frame")

    no = expansion.region.material.n
    ni = image_region.material.n
    qu, qv, ci, ct, support, gain = _pupil_geometry(
        grid, expansion.wavelength0, no, ni, magnification, numerical_aperture
    )
    normal = expansion.wavevectors @ grid.frame.normal
    occupied = np.any(expansion.amplitudes != 0, axis=-1)

    if np.any(normal.real[occupied] < 0):
        raise ValueError("ideal imaging requires forward incident modes")

    radial = np.hypot(qu, qv)
    azimuth = np.arctan2(qv, qu)
    azimuth = np.where(radial == 0, 0, azimuth)
    s = np.stack((-np.sin(azimuth), np.cos(azimuth), np.zeros_like(qu)), axis=-1)
    incident = np.stack((qu / no, qv / no, ci), axis=-1)
    outgoing = np.stack(
        (qu / (magnification * ni), qv / (magnification * ni), ct), axis=-1
    )
    p_incident = np.cross(s, incident)
    p_outgoing = np.cross(s, outgoing)
    local = expansion.amplitudes.reshape(grid.shape + (3,)) @ grid.frame.basis
    components = np.stack(
        (np.sum(local * p_incident, axis=-1), np.sum(local * s, axis=-1)), axis=-1
    )

    if pupil is not None:
        pupil = np.asarray(pupil, dtype=complex)

        if not np.isfinite(pupil).all():
            raise ValueError("pupil must be finite")

        if pupil.shape == grid.shape + (2, 2) or pupil.shape == (2, 2):
            components = np.einsum("...ij,...j->...i", pupil, components)
        elif pupil.ndim == 0 or pupil.shape == grid.shape:
            components = components * np.broadcast_to(pupil, grid.shape)[..., None]
        else:
            raise ValueError("pupil must be scalar, grid-shaped, or a Jones matrix")

    local_output = (components[..., :1] * p_outgoing + components[..., 1:] * s) * gain[
        ..., None
    ]
    local_output[~support] = 0
    local_output = _invert_frequencies(local_output, magnification)

    # Outside the passed pupil, choose valid empty forward bins for FFT synthesis.
    fx, fy = frequency_grid(grid.coordinates)
    k0 = 2 * np.pi / expansion.wavelength0
    ku = 2 * np.pi * fx / abs(magnification)
    kv = 2 * np.pi * fy / abs(magnification)
    kn = np.sqrt(((k0 * ni) ** 2 - ku**2 - kv**2).astype(complex))
    wavevectors = np.stack((ku, kv, kn), axis=-1) @ image_frame.basis.T
    center_u = grid.u[grid.shape[1] // 2]
    center_v = grid.v[grid.shape[0] // 2]
    output_grid = CartesianGrid(
        abs(magnification) * (grid.u - center_u) + magnification * center_u,
        abs(magnification) * (grid.v - center_v) + magnification * center_v,
        image_frame,
    )
    origin = output_grid.points[grid.shape[0] // 2, grid.shape[1] // 2]
    output = PlaneWaveExpansion(
        wavevectors.reshape(-1, 3),
        (local_output @ image_frame.basis.T).reshape(-1, 3),
        origin,
        expansion.wavelength0,
        image_region,
    )

    return PlaneSpectrum(output_grid, output)


def scalar_image(
    amplitude,
    grid,
    wavelength0,
    object_region,
    image_region,
    *,
    magnification,
    numerical_aperture,
    pupil=1,
    defocus=0,
):
    """Scalar reference on the same scaled grid as image_spectrum.

    Uses the same aperture, obliquity/Jacobian and exact defocus dispersion, but
    no polarization transport. Returns a complex array, not a Maxwell E/H field.
    This separates the scalar approximation from a paraxial propagation error.
    """
    amplitude = np.asarray(amplitude, dtype=complex)
    pupil = np.asarray(pupil, dtype=complex)

    if amplitude.shape != grid.shape or not np.isfinite(amplitude).all():
        raise ValueError("scalar amplitude must be finite and grid-shaped")

    if pupil.shape not in ((), grid.shape) or not np.isfinite(pupil).all():
        raise ValueError("scalar pupil must be finite and scalar or grid-shaped")

    if not np.isfinite(defocus):
        raise ValueError("defocus must be finite")

    _, _, _, ct, _, gain = _pupil_geometry(
        grid,
        wavelength0,
        object_region.material.n,
        image_region.material.n,
        magnification,
        numerical_aperture,
    )
    phase = np.exp(2j * np.pi * image_region.material.n * ct * defocus / wavelength0)
    modes = FFT2(amplitude) * pupil * gain * phase

    return IFFT2(_invert_frequencies(modes, magnification))


def apply_pupil(source, pupil):
    """Apply a Jones pupil to forward propagating modes in their local (p, s) basis.

    The pupil is a prescribed angular response, not a material surface inferred
    from a sampled mask. This function can prepare illumination or model an ideal
    pupil element. H follows from the unchanged mode wavevectors.
    """
    if not isinstance(source, PlaneSpectrum):
        raise TypeError("source must be a PlaneSpectrum")

    expansion = source.expansion
    grid = source.grid
    wavevectors = expansion.wavevectors @ grid.frame.basis
    occupied = np.any(expansion.amplitudes != 0, axis=-1)

    if np.any(abs(wavevectors.imag[occupied]) > 1e-10) or np.any(
        wavevectors.real[occupied, 2] <= 0
    ):
        raise ValueError("Jones pupil requires forward propagating occupied modes")

    direction = wavevectors.real / (
        2 * np.pi * expansion.region.material.n / expansion.wavelength0
    )
    azimuth = np.arctan2(direction[:, 1], direction[:, 0])
    s = np.column_stack((-np.sin(azimuth), np.cos(azimuth), np.zeros(len(azimuth))))
    p = np.cross(s, direction)
    local = expansion.amplitudes @ grid.frame.basis
    components = np.stack(
        (np.sum(local * p, axis=-1), np.sum(local * s, axis=-1)), axis=-1
    )
    pupil = np.asarray(pupil, dtype=complex)

    if pupil.shape not in ((2, 2), grid.shape + (2, 2)) or not np.isfinite(pupil).all():
        raise ValueError(
            "pupil must be a finite 2 x 2 matrix or a grid of such matrices"
        )

    matrices = np.broadcast_to(pupil, grid.shape + (2, 2)).reshape(-1, 2, 2)
    components = np.einsum("nij,nj->ni", matrices, components)
    values = components[:, :1] * p + components[:, 1:] * s
    values[~occupied] = 0
    output = PlaneWaveExpansion(
        expansion.wavevectors,
        values @ grid.frame.basis.T,
        expansion.origin,
        expansion.wavelength0,
        expansion.region,
    )

    return PlaneSpectrum(grid, output)


def focus_pupil(
    field, object_region, image_region, *, focal_length, numerical_aperture
):
    """Focus a collimated transverse pupil with an ideal aplanatic objective.

    The sine-condition pupil coordinate is rho = f sin(theta). Uniform input
    rays travel along the grid normal. The output is a complete vector spectrum
    about grid.origin + f * normal; its FFT spacing follows the physical pupil.
    This ideal objective includes polarization rotation and the power Jacobian,
    not dielectric Fresnel losses or a particular lens prescription.
    """
    from ..fields import TransverseElectricField

    if not isinstance(field, TransverseElectricField) or not isinstance(
        field.grid, CartesianGrid
    ):
        raise TypeError("focus_pupil requires a Cartesian transverse pupil field")

    f = positive(focal_length, "focal_length")
    na = positive(numerical_aperture, "numerical_aperture")
    grid = field.grid
    no, ni = object_region.material.n, image_region.material.n

    if na >= ni:
        raise ValueError("the objective must stay below grazing")

    ny, nx = grid.shape
    if abs(grid.u[nx // 2]) > 1e-15 or abs(grid.v[ny // 2]) > 1e-15:
        raise ValueError("pupil coordinates must use the centered FFT lattice")

    x, y = np.meshgrid(grid.u, grid.v)
    radial = np.hypot(x, y)
    support = radial <= f * na / ni
    cosine = np.sqrt(np.maximum(0, 1 - (radial / f) ** 2))
    azimuth = np.arctan2(y, x)
    s = np.stack((-np.sin(azimuth), np.cos(azimuth), np.zeros_like(x)), axis=-1)
    p_incident = np.stack((np.cos(azimuth), np.sin(azimuth), np.zeros_like(x)), axis=-1)
    direction = np.stack((-x / f, -y / f, cosine), axis=-1)
    p_outgoing = np.cross(s, direction)
    local = np.concatenate((field.values, np.zeros(grid.shape + (1,))), axis=-1)
    ep = np.sum(local * p_incident, axis=-1)
    es = np.sum(local * s, axis=-1)
    gain = np.zeros(grid.shape)
    gain[support] = np.sqrt(no / (ni * cosine[support]))
    du, dv = grid.spacing
    fourier_scale = du * dv * ni / (field.wavelength0 * f)
    modes = (
        (ep[..., None] * p_outgoing + es[..., None] * s)
        * gain[..., None]
        * fourier_scale
    )
    modes = _invert_frequencies(modes, -1)
    spacing_u = field.wavelength0 * f / (ni * nx * du)
    spacing_v = field.wavelength0 * f / (ni * ny * dv)
    frame = Frame(grid.frame.origin + f * grid.frame.normal, grid.frame.basis)
    image_grid = CartesianGrid(
        (np.arange(nx) - nx // 2) * spacing_u,
        (np.arange(ny) - ny // 2) * spacing_v,
        frame,
    )
    fx, fy = frequency_grid(image_grid.coordinates)
    ku, kv = 2 * np.pi * fx, 2 * np.pi * fy
    kn = np.sqrt(
        ((2 * np.pi * ni / field.wavelength0) ** 2 - ku**2 - kv**2).astype(complex)
    )
    vectors = np.stack((ku, kv, kn), axis=-1) @ frame.basis.T
    expansion = PlaneWaveExpansion(
        vectors.reshape(-1, 3),
        (modes @ frame.basis.T).reshape(-1, 3),
        frame.origin,
        field.wavelength0,
        image_region,
    )

    return PlaneSpectrum(image_grid, expansion)
