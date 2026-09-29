"""Forward optical paths composed as R Q^-1 T Q R at every interface."""

from dataclasses import dataclass

import numpy as np
import scipy.fft

from ..fields import ElectricField, ElectromagneticField, TransverseElectricField
from ..fields.eikonal import EikonalField
from ..fields.spectrum import PlaneSpectrum, PlaneWaveExpansion
from ..fourier.cartesian import FFT2, IFFT2, frequency_grid
from ..sampling import CartesianGrid
from ..surfaces import FreeformSurface
from .angular_spectrum import angular_spectrum
from .operators import PlanarASM, SurfaceTransport, fresnel_transmission


@dataclass(frozen=True)
class HighFrequencySolution:
    """Outgoing homogeneous spectra; evaluate uses ASM/plane-wave synthesis only.

    interface_spectra[mode][interface] retains each Q^-1 result for physical
    diagnostics, including interfaces before the final observation region.
    interface_spectra[mode][i] is interface i's own Q^-1 result.
    exterior_spectra[mode] holds, for each element with a finite edge, the
    light that passed outside it, at the element's exit vertex plane. It is
    added to the field sent onward, so it is included in transmitted.
    coupled_elements lists entrance/exit interface pairs that used the
    approximate two-medium rim propagation.
    """

    transmitted: tuple[PlaneSpectrum, ...]
    encounters: tuple[tuple, ...]
    inverse_reports: tuple[tuple, ...]
    transports: tuple[tuple, ...]
    interface_spectra: tuple[tuple[PlaneSpectrum, ...], ...]
    exterior_spectra: tuple[tuple[PlaneSpectrum, ...], ...] = ()
    coupled_elements: tuple[tuple[int, int], ...] = ()

    def evaluate(self, grid, *, backend="auto", **kwargs):
        fields = []
        for spectrum in self.transmitted:
            plane = spectrum.grid
            displacement = grid.points - plane.frame.origin
            boundary = max(0.0, float(self.transports[-1][1].height.max()))
            if np.any(displacement @ plane.frame.normal < boundary - 1e-14):
                raise ValueError(
                    "observation must be beyond the final surface envelope and reference plane"
                )
            matching = (
                isinstance(grid, CartesianGrid)
                and grid.shape == plane.shape
                and np.array_equal(grid.u, plane.u)
                and np.array_equal(grid.v, plane.v)
                and np.allclose(grid.frame.basis, plane.frame.basis, atol=1e-12, rtol=0)
            )
            delta = grid.frame.origin - plane.frame.origin if matching else None
            if matching and np.linalg.norm(delta @ plane.frame.basis[:, :2]) < 1e-14:
                value = PlanarASM()(spectrum, float(delta @ plane.frame.normal))
                fields.append(
                    ElectromagneticField(
                        ElectricField(
                            grid, value.electric.wavelength0, value.electric.values
                        ),
                        value.magnetic,
                    )
                )
            else:
                selected = backend
                if selected == "auto":
                    try:
                        import finufft  # noqa: F401

                        selected = "nufft"
                    except ImportError:
                        selected = "direct"
                fields.append(spectrum.evaluate(grid, backend=selected, **kwargs))
        return ElectromagneticField(
            ElectricField(
                grid,
                fields[0].electric.wavelength0,
                sum(f.electric.values for f in fields),
            ),
            sum(f.magnetic for f in fields),
        )

    def electric(self, grid, **kwargs):
        return self.evaluate(grid, **kwargs).electric

    def evaluate_green_comparison(self, grid, **kwargs):
        """Separate physical-optics comparison, never used by evaluate()."""
        from ..comparisons.surface_radiation import SurfaceRadiation

        fields = [
            SurfaceRadiation.from_field(
                mode[-1].transmitted, spectrum.expansion.region
            ).evaluate(grid, **kwargs)
            for mode, spectrum in zip(self.encounters, self.transmitted)
        ]
        return ElectromagneticField(
            ElectricField(
                grid,
                fields[0].electric.wavelength0,
                sum(f.electric.values for f in fields),
            ),
            sum(f.magnetic for f in fields),
        )


def _branches(source):
    if isinstance(source, EikonalField):
        return (source,)
    if isinstance(source, PlaneSpectrum):
        return (source,)  # caller declares one locally eikonal sampled branch
    if not isinstance(source, PlaneWaveExpansion):
        raise TypeError(
            "source must declare an EikonalField or analytic plane-wave branches"
        )
    result = []
    for k, amplitude in zip(source.wavevectors, source.amplitudes):
        if not np.any(amplitude):
            continue
        if np.any(k.imag):
            raise ValueError("incident eikonal branches must be propagating")
        gradient = k.real * source.wavelength0 / (2 * np.pi)
        result.append(
            EikonalField(
                lambda points, g=gradient: (points - source.origin) @ g,
                lambda points, g=gradient: g,
                lambda points, a=amplitude: a,
                source.wavelength0,
                source.region,
            )
        )
    if not result:
        raise ValueError("source needs a nonzero phase branch")
    return tuple(result)


def _masked(field, mask):
    return ElectromagneticField(
        ElectricField(
            field.electric.grid,
            field.electric.wavelength0,
            field.electric.values * mask[..., None],
        ),
        field.magnetic * mask[..., None],
    )


def _add_spectra(a, b):
    """Coherent sum of two FFT spectra on the same plane lattice and medium."""
    if (
        a.expansion.region.material != b.expansion.region.material
        or a.grid.shape != b.grid.shape
        or not np.allclose(
            a.grid.points, b.grid.points, rtol=0, atol=1e-12 * np.ptp(a.grid.points)
        )
        or not np.allclose(a.expansion.wavevectors, b.expansion.wavevectors)
    ):
        raise ValueError("spectra must share their plane lattice and medium")
    k = a.expansion.wavevectors
    total = a.expansion.amplitudes + b.expansion.amplitudes
    # Both inputs are transverse; strong cancellation (e.g. an index-matched
    # element) leaves only a roundoff longitudinal part, removed with the
    # bilinear projection that holds for complex (evanescent) k as well.
    total = total - k * (np.sum(k * total, -1) / np.sum(k * k, -1))[:, None]
    return PlaneSpectrum(
        a.grid,
        PlaneWaveExpansion(
            k,
            total,
            a.expansion.origin,
            a.expansion.wavelength0,
            a.expansion.region,
        ),
    )


def _elements(system):
    """Elements as (first, last, enclosed) interface index triples.

    The ambient material is the source medium. An element is a run of
    interfaces that starts where light leaves the ambient material and ends
    where it returns to it (enclosed). A trailing run that never returns is
    open: beyond a finite edge there is no defined medium for the field to
    continue in, so its entrance edge must carry a stop that covers it.
    """
    ambient = system.interfaces[0].minus.material
    elements, start = [], None
    for index, interface in enumerate(system.interfaces):
        if start is None:
            start = index
        if interface.plus.material == ambient:
            elements.append((start, index, True))
            start = None
    if start is not None:
        elements.append((start, len(system.interfaces) - 1, False))
    for first, last, enclosed in elements:
        edge, stop = system.apertures[first], system.stops[first]
        if not enclosed and edge is not None and (stop is None or stop > edge):
            raise ValueError(
                "an open system (no return to the source medium) with a finite "
                "surface edge needs a stop no larger than that edge; otherwise "
                "the field beyond the edge has no medium to continue in"
            )
        for inner in system.apertures[first + 1 : last + 1]:
            if edge is not None and (inner is None or inner > edge):
                raise ValueError(
                    "a surface edge inside an element exceeds the entrance edge; "
                    "light passing outside the element would meet that face"
                )
    return elements


def _chart_surface(surface, aperture):
    """Constant-height extension of a surface beyond its edge.

    Only the disk inside the edge is a material face. The flat extension at
    rim height closes the periodic numerical chart; what happens there (field
    passing outside the element, sidewall leakage, or a stop) is decided by
    the propagation, not by the chart.
    """
    if aperture is None:
        return surface

    def coordinates(xy):
        radius = np.linalg.norm(xy, axis=-1)
        return xy * np.minimum(1, aperture / np.maximum(radius, 1e-300))[..., None]

    def gradient(xy):
        # Outside: h(a*xy/r), including angular variation for a freeform rim.
        radius = np.linalg.norm(xy, axis=-1)
        unit = xy / np.maximum(radius, 1e-300)[..., None]
        grad = surface.gradient(coordinates(xy))
        exterior = (
            aperture
            / np.maximum(radius, 1e-300)[..., None]
            * (grad - np.sum(grad * unit, axis=-1)[..., None] * unit)
        )
        return np.where((radius <= aperture)[..., None], grad, exterior)

    return FreeformSurface(
        lambda xy: surface.height(coordinates(xy)), gradient, surface.frame
    )


def _local_directions(spectrum, transport, illuminated):
    """Local phase gradient Im(E* . grad E)/|E|² within one WKB branch.

    Spectral differentiation avoids phase unwrapping and works across 2pi cuts.
    This is a local eikonal estimate, not a ray assignment to a coherent sum of
    independent branches. Dark samples use the forward axis and carry no flux.
    """
    coefficients = (
        spectrum.expansion.amplitudes.reshape(transport.plane.shape + (3,))
        * transport.size
    )
    electric = transport._synthesize(coefficients)
    intensity = np.sum(abs(electric) ** 2, axis=-1)
    active = illuminated & (intensity > intensity.max() * 1e-12)
    wavevectors = spectrum.expansion.wavevectors.reshape(coefficients.shape)
    gradient = np.stack(
        [
            np.imag(
                np.sum(
                    electric.conj()
                    * transport._synthesize(
                        1j * wavevectors[..., axis, None] * coefficients
                    ),
                    axis=-1,
                )
            )
            for axis in range(3)
        ],
        axis=-1,
    )
    length = np.linalg.norm(gradient, axis=-1)
    directions = np.broadcast_to(transport.plane.frame.normal, electric.shape).copy()
    valid = active & (length > 0)
    directions[valid] = gradient[valid] / length[valid, None]
    return directions


def _rim_planes(system, start, end, wavelength0):
    """Return the two flat rim planes of an axisymmetric, two-face element."""
    if end != start + 1 or any(system.stops[start : end + 1]):
        return None
    a = system.apertures[start]
    if a is None or system.apertures[end] != a:
        return None
    first, last = system.interfaces[start], system.interfaces[end]
    if np.isclose(first.plus.material.n, first.minus.material.n):
        return None
    azimuth = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    rim = a * np.column_stack((np.cos(azimuth), np.sin(azimuth)))
    h1, h2 = first.surface.height(rim), last.surface.height(rim)
    if np.ptp(h1) > 1e-10 * wavelength0 or np.ptp(h2) > 1e-10 * wavelength0:
        return None
    z1 = float(h1[0])
    separation = float(
        (last.surface.frame.origin - first.surface.frame.origin)
        @ first.surface.frame.normal
    )
    z2 = separation + float(h2[0])
    if z2 <= z1:
        return None
    return z1, z2


def _rim_band(values, plane, region, wavelength0, max_sine):
    """Re-expand a two-component plane field into forward Maxwell modes."""
    fx, fy = frequency_grid(plane.coordinates)
    k = 2 * np.pi * region.material.n / wavelength0
    keep = fx**2 + fy**2 <= (max_sine * k / (2 * np.pi)) ** 2
    limited = IFFT2(FFT2(values) * keep[..., None])
    return angular_spectrum(
        TransverseElectricField(plane, wavelength0, limited),
        region,
        evanescent="discard",
    )


def _couple_rim(inside, outside, aperture, z1, z2, *, max_sine, step):
    """Forward split-step coupling between glass and air across a finite rim.

    This is a two-medium approximation. It propagates in each homogeneous
    medium exactly per slice, then exchanges transverse E at the radial mask.
    The sqrt(index ratio) factors preserve the normal-incidence flux scale;
    they do not impose the complete vector sidewall boundary conditions.
    """
    plane = inside.grid
    lam = inside.expansion.wavelength0
    inner, outer = inside.expansion.region, outside.expansion.region
    n_in, n_out = inner.material.n, outer.material.n
    xy = (plane.points - plane.frame.origin) @ plane.frame.basis[:, :2]
    clear = np.linalg.norm(xy, axis=-1) <= aperture
    mask = clear[..., None]
    asm = PlanarASM()
    e_in = asm(inside, z1).electric.values @ plane.frame.basis[:, :2]
    e_out = asm(outside, z1).electric.values @ plane.frame.basis[:, :2]
    in_from_out = np.sqrt(n_out / n_in)
    out_from_in = np.sqrt(n_in / n_out)
    field = np.where(mask, e_in + in_from_out * e_out, e_out + out_from_in * e_in)

    slices = max(1, int(np.ceil((z2 - z1) / (step * lam))))
    if slices > 4096:
        raise ValueError("rim_step requests more than 4096 axial slices")
    dz = (z2 - z1) / slices
    fx, fy = frequency_grid(plane.coordinates)
    frequency2 = (2 * np.pi) ** 2 * (fx**2 + fy**2)
    phases = [
        np.fft.ifftshift(
            np.exp(
                1j
                * np.sqrt(((2 * np.pi * n / lam) ** 2 - frequency2).astype(complex))
                * dz
            )
        )
        for n in (n_in, n_out)
    ]
    for _ in range(slices):
        from_in = np.where(mask, field, 0)
        from_out = np.where(mask, 0, field)
        through_in = scipy.fft.ifft2(
            scipy.fft.fft2(from_in + in_from_out * from_out, axes=(0, 1), workers=-1)
            * phases[0][..., None],
            axes=(0, 1),
            workers=-1,
        )
        through_out = scipy.fft.ifft2(
            scipy.fft.fft2(from_out + out_from_in * from_in, axes=(0, 1), workers=-1)
            * phases[1][..., None],
            axes=(0, 1),
            workers=-1,
        )
        field = np.where(mask, through_in, through_out)

    # plane is the entrance vertex, while field is currently at z2.
    exit_plane = plane.translated(z2 * plane.frame.normal)
    in_spec = _rim_band(np.where(mask, field, 0), exit_plane, inner, lam, max_sine)
    out_spec = _rim_band(np.where(mask, 0, field), exit_plane, outer, lam, 0.999)
    return (
        asm.spectrum(in_spec, -z2),
        asm.spectrum(out_spec, -z2),
    )


def propagate_high_frequency(
    source,
    system,
    grid,
    *,
    max_sine=0.85,
    transport_tolerance=1e-10,
    inverse_tolerance=1e-9,
    max_iterations=200,
    rim_coupling="auto",
    rim_step=0.05,
):
    """Compose Q -> T -> Q^-1, then R to each following vertex plane.

    grid is a padded Cartesian tangent-plane lattice at the first vertex.
    Interface frames must share its orientation and transverse origin. Each
    analytic incident phase is transported independently and summed coherently.
    A PlaneSpectrum declares one locally eikonal branch; do not use a coherent
    mixture with multiple local wavevectors as a single eikonal input.
    Surface edges (system.apertures) never cut the field; pupils
    (system.stops) do. An element is a run of interfaces between visits to the
    source medium. At an element's entrance edge the field inside the edge
    enters the element; the field outside it continues in the surrounding
    medium: it is inverted there on the entrance chart (flat beyond the rim),
    carried across the element by exact ASM and added at the element's exit.
    Inside an element, light reaching a later chart beyond that surface's edge
    crosses the chart's flat extension at rim height with normal-incidence
    Fresnel, a simple model of leaving through the sidewall (exact for an
    index-matched element). A stop cuts everything on its chart outside its
    radius, including light passing outside the element. Sidewall reflection
    and light entering through a sidewall are not modeled by the uncoupled
    path. For a thick, two-face element with a circular rim and no stop,
    rim_coupling='auto' uses a forward two-medium slice approximation when
    the diameter is between four and ten wavelengths, its centre thickness
    is at most four wavelengths, and
    a²/(wavelength × centre thickness) <= 8. 'off' retains the original
    separated path; 'on' requests coupling for every supported element.
    Neither option solves reflected sidewall or returning branches. The user
    must resolve phase, curvature and edges and
    verify both FFT-cell padding and grid convergence. No Green kernel is used.

    """
    if not isinstance(grid, CartesianGrid):
        raise TypeError(
            "operational propagation requires a padded CartesianGrid, not surface quadrature"
        )
    if rim_coupling not in ("auto", "off", "on"):
        raise ValueError("rim_coupling must be 'auto', 'off', or 'on'")
    if not np.isfinite(rim_step) or rim_step <= 0:
        raise ValueError("rim_step must be a positive finite fraction of wavelength")
    first = system.interfaces[0]
    context = source.expansion if isinstance(source, PlaneSpectrum) else source
    if context.region != first.minus:
        raise ValueError("source region does not match the first incident medium")
    if not np.allclose(
        grid.frame.origin, first.surface.frame.origin, atol=1e-14, rtol=0
    ):
        raise ValueError("input grid must be at the first interface vertex")
    positions = []
    for interface in system.interfaces:
        frame = interface.surface.frame
        if not np.allclose(frame.basis, grid.frame.basis, atol=1e-12, rtol=0):
            raise ValueError("operational path requires parallel chart frames")
        delta = (frame.origin - grid.frame.origin) @ grid.frame.basis
        if np.linalg.norm(delta[:2]) > 1e-14:
            raise ValueError("operational path requires aligned chart origins")
        positions.append(delta[2])
    if np.any(np.diff(positions) <= 0):
        raise ValueError("interface vertices must be in forward order")

    starts, ends, coupled = set(), {}, {}
    for start, end, enclosed in _elements(system):
        if enclosed:
            starts.add(start)
            ends[end] = start
            if rim_coupling != "off":
                planes = _rim_planes(system, start, end, context.wavelength0)
                if planes is not None:
                    aperture = system.apertures[start]
                    centre_thickness = positions[end] - positions[start]
                    fresnel_number = aperture**2 / (
                        context.wavelength0 * centre_thickness
                    )
                    diameter = 2 * aperture / context.wavelength0
                    if rim_coupling == "on" or (
                        4 <= diameter <= 10
                        and centre_thickness <= 4 * context.wavelength0
                        and fresnel_number <= 8
                    ):
                        coupled[start] = (end, *planes)
                elif (
                    rim_coupling == "on"
                    and system.apertures[start] is not None
                    and not np.isclose(
                        system.interfaces[start].minus.material.n,
                        system.interfaces[start].plus.material.n,
                    )
                ):
                    raise ValueError(
                        "rim coupling requires two axisymmetric faces with "
                        "equal circular edges and no stops"
                    )
    coupled_exits = {end for end, *_ in coupled.values()}
    ambient = first.minus

    # Geometry and Q factorizations are shared by all independent phases.
    operators = []
    for interface, aperture in zip(system.interfaces, system.apertures):
        plane = CartesianGrid(grid.u, grid.v, interface.surface.frame)
        surface = _chart_surface(interface.surface, aperture)
        options = dict(max_sine=max_sine, tolerance=transport_tolerance)
        operators.append(
            (
                SurfaceTransport(
                    plane, surface, interface.minus, context.wavelength0, **options
                ),
                SurfaceTransport(
                    plane, surface, interface.plus, context.wavelength0, **options
                ),
            )
        )
    # Light outside an element travels in the source medium; a stop inside an
    # element must also act on it, on that element's chart.
    outside_ops = {
        index: SurfaceTransport(
            CartesianGrid(grid.u, grid.v, interface.surface.frame),
            _chart_surface(interface.surface, aperture),
            ambient,
            context.wavelength0,
            max_sine=max_sine,
            tolerance=transport_tolerance,
        )
        for index, (interface, aperture, stop) in enumerate(
            zip(system.interfaces, system.apertures, system.stops)
        )
        if stop is not None and index not in starts
    }
    for previous, following in zip(operators, operators[1:]):
        if np.any(
            (following[0].grid.points - previous[0].grid.points) @ grid.frame.normal
            <= 0
        ):
            raise ValueError("successive surfaces intersect on the numerical chart")

    outgoing, all_encounters, all_reports, all_spectra = [], [], [], []
    all_exterior = []
    branches = _branches(source)
    for branch in branches:
        current = None
        encounters, reports, spectra, exteriors = [], [], [], []
        pending = None
        for index, (interface, aperture, stop, (Q_in, Q_out)) in enumerate(
            zip(system.interfaces, system.apertures, system.stops, operators)
        ):
            radius = np.linalg.norm(Q_in.xy, axis=-1)
            clear = (
                np.ones(grid.shape, bool) if aperture is None else radius <= aperture
            )
            pupil = np.ones(grid.shape, bool) if stop is None else radius <= stop
            if current is None and isinstance(branch, EikonalField):
                incident = branch.evaluate(Q_in.grid)  # analytic Q R
                directions = branch.directions(Q_in.grid.points)
            elif current is None:
                incident = Q_in.forward(branch)
                directions = _local_directions(branch, Q_in, clear & pupil)
            else:
                # R_xi in the intervening homogeneous medium.
                current = PlanarASM().spectrum(
                    current, positions[index] - positions[index - 1]
                )
                incident = Q_in.forward(current)
                directions = _local_directions(current, Q_in, clear & pupil)
            if stop is not None and pending is not None:
                # The pupil also blocks light passing outside the element.
                carried = PlanarASM().spectrum(
                    pending[0], positions[index] - pending[1]
                )
                Q_out_side = outside_ops[index]
                trace = Q_out_side.forward(carried)
                pending = (
                    Q_out_side.inverse(
                        _masked(trace, pupil),
                        tolerance=inverse_tolerance,
                        max_iterations=max_iterations,
                    )[0],
                    positions[index],
                )
            incident = _masked(incident, pupil)
            if index in starts and aperture is not None:
                # Beyond the entrance edge the field stays in the source medium.
                outside = _masked(incident, ~clear)
                if np.any(outside.electric.values):
                    pending = (
                        Q_in.inverse(
                            outside,
                            tolerance=inverse_tolerance,
                            max_iterations=max_iterations,
                        )[0],
                        positions[index],
                    )
                incident = _masked(incident, clear)
            # Beyond a surface edge there is no optical face: the chart's flat
            # extension is crossed at normal incidence, because local phase
            # directions of weak diffraction tails are not meaningful rays.
            directions = np.where(clear[..., None], directions, Q_in.grid.normals)
            if index in coupled_exits:
                # Rim diffraction may create exit-face TIR, which the forward
                # interface operator cannot represent. Keep the supported
                # transmitted branch and discard those samples explicitly.
                cosine = np.abs(np.sum(directions * Q_in.grid.normals, axis=-1))
                sine2 = np.maximum(0, 1 - cosine**2)
                total_reflection = clear & (
                    interface.minus.material.n**2 * sine2
                    >= interface.plus.material.n**2
                )
                incident = _masked(incident, ~total_reflection)
                directions = np.where(
                    total_reflection[..., None], Q_in.grid.normals, directions
                )
            encounter = fresnel_transmission(incident, interface, directions)  # T
            current, report = Q_out.inverse(
                encounter.transmitted,
                tolerance=inverse_tolerance,
                max_iterations=max_iterations,
            )  # Q^-1
            encounters.append(encounter)
            reports.append(report)
            spectra.append(current)  # this interface's own Q^-1 result
            if index in coupled and pending is not None:
                _, z1, z2 = coupled[index]
                current, exterior = _couple_rim(
                    current,
                    pending[0],
                    system.apertures[index],
                    z1,
                    z2,
                    max_sine=max_sine,
                    step=rim_step,
                )
                pending = (exterior, positions[index])
            if index in ends and pending is not None:
                carried = PlanarASM().spectrum(
                    pending[0], positions[index] - pending[1]
                )
                exteriors.append(carried)
                current = _add_spectra(current, carried)
                pending = None
        outgoing.append(current)
        all_encounters.append(tuple(encounters))
        all_reports.append(tuple(reports))
        all_spectra.append(tuple(spectra))
        all_exterior.append(tuple(exteriors))
    return HighFrequencySolution(
        tuple(outgoing),
        tuple(all_encounters),
        tuple(all_reports),
        tuple(operators),
        tuple(all_spectra),
        tuple(all_exterior),
        tuple((start, end) for start, (end, *_rest) in coupled.items()),
    )
