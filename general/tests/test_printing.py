"""The printed form of every public object: short reprs, organized strs, tables.

Every public class is printed from an instance built here (the zoo): its repr
is one bounded line naming its defining numbers, without arrays or addresses,
and neither repr nor str raises. The optical system prints as a prescription
table, in text, HTML and as data; the Solution as a table of its reports.
"""

import inspect
from html.parser import HTMLParser

import numpy as np
import pytest
import vecdiff as o
import vecdiff.references
from vecdiff._printing import guarded, number, text_table
from vecdiff.catalog import thorlabs_al1225

F, S, P, Y = o.fields, o.surfaces, o.propagation, o.system
Frame = o.geometry.Frame
AIR = o.media.HomogeneousRegion("air", o.media.air)
GLASS = o.media.HomogeneousRegion("glass", o.media.Dielectric(1.5, "glass"))
C = 1 / np.sqrt(2)
TILTED = np.array([[1, 0, 0], [0, C, C], [0, -C, C]]).T


def lens():
    """The plano-convex lens of the example: stop, flat face, curved face."""
    bk7 = o.media.Dielectric(1.5, "BK7-like")
    return Y.OpticalSystem(
        Y.Stop(4e-3, z=-0.5e-3),
        Y.Refraction(S.Plane.z(0.0), o.media.air, bk7),
        Y.Refraction(S.SphericalCap(-25e-3, Frame([0, 0, 3e-3])), bk7, o.media.air),
    )


def ghost(thickness=100.0):
    glass = GLASS.material
    front, back = S.Plane.z(0.0), S.Plane.z(thickness)
    return Y.OpticalSystem(
        Y.Refraction(front, o.media.air, glass),
        Y.Reflection(back, glass, behind=o.media.air),
        Y.Reflection(front, glass, behind=o.media.air),
        Y.Refraction(back, glass, o.media.air),
    )


def cassegrain():
    primary = S.EvenAsphere(-0.01, conic=-1.0)
    secondary = S.EvenAsphere(-1 / 24, conic=-1.96, frame=Frame([0, 0, -40.0]))
    return Y.OpticalSystem(
        Y.Stop(36.0, z=-10.0),
        Y.Reflection(primary, o.media.air),
        Y.Reflection(secondary, o.media.air),
    )


def fold():
    return Y.OpticalSystem(Y.Reflection(S.Plane(Frame([0, 0, 0], TILTED)), AIR))


def systems():
    ball = S.Sphere(10.0, Frame([0, 0, -10.0]))
    cylinder = S.Cylinder(20.0, Frame([0, 0, -20.0]))
    room = o.media.HomogeneousRegion("room", o.media.air)
    return dict(
        lens=lens(),
        ghost=ghost(),
        cassegrain=cassegrain(),
        fold=fold(),
        ball=Y.OpticalSystem(
            Y.Stop(8.0, z=-10.5),
            Y.Refraction(ball, AIR, GLASS),
            Y.Reflection(ball, GLASS, behind=AIR),
            Y.Refraction(ball, GLASS, AIR),
        ),
        cylinder=Y.OpticalSystem(
            Y.Refraction(cylinder, AIR, GLASS),
            Y.Refraction(cylinder.far_side(), GLASS, AIR),
        ),
        regions=Y.OpticalSystem(
            Y.Refraction(S.Plane.z(0), room, GLASS, edge=12.5),
            Y.Refraction(
                S.CartesianOval(-np.inf, 30, 1.5, 1.0, Frame([0, 0, 5])),
                GLASS,
                o.media.HomogeneousRegion("detector side", o.media.air),
            ),
        ),
        freeform=Y.OpticalSystem(
            Y.Refraction(
                S.FreeformSurface(
                    lambda xy: 0 * xy[..., 0], lambda xy: 0 * xy, Frame([0.5, 0, 0])
                ),
                AIR,
                GLASS,
            ),
            Y.Refraction(
                S.ReversedSurface(S.CartesianOval(-np.inf, 30, 1, 1.5)), GLASS, AIR
            ),
        ),
        al1225=thorlabs_al1225(),
        empty=Y.OpticalSystem(medium=o.media.air),
        bare=Y.OpticalSystem(),
    )


def lattice(half, step=0.4):
    count = int(2 * half / step) // 2 * 2
    axis = (np.arange(count) - count // 2) * step
    return o.sampling.CartesianGrid(axis, axis)


def wave_problem():
    axis = np.arange(-16, 16) / 8
    grid = o.sampling.CartesianGrid(axis, axis, Frame([0, 0, -1.0]))
    after = o.media.HomogeneousRegion("after", o.media.Dielectric(1))
    front = S.FreeformSurface(
        lambda p: 0.2 * np.cos(p[..., 0]),
        lambda p: np.stack((-0.2 * np.sin(p[..., 0]), np.zeros_like(p[..., 0])), -1),
    )
    system = Y.OpticalSystem(
        Y.Refraction(front, AIR, GLASS),
        Y.Refraction(S.Plane.z(0.8), GLASS, after),
    )
    values = np.zeros(grid.shape + (2,), complex)
    values[..., 0] = 1
    return F.TransverseElectricField(grid, 1, values), system


def build_zoo():
    """One instance of every public class, and some degenerate ones."""
    zoo = {f"system {k}": v for k, v in systems().items()}
    focused = F.FocusedBeam(
        F.profiles.Gaussian(0.3),
        F.polarization.x,
        focal_length=1.0,
        max_sine=0.4,
        wavelength0=1.0,
        region=AIR,
        frame=Frame([0, 0, 15.0]),
    )
    cap = S.SphericalCap(10.0, Frame([0, 0, -10.0]))
    meridian = P.Propagator(Y.Refraction(cap, AIR, GLASS))
    solution = meridian(focused)
    plane = F.PlaneBeam(
        F.profiles.Gaussian(3.0),
        F.polarization.y,
        wavelength0=1.0,
        region=AIR,
        frame=Frame([0, 0, -5.0]),
    )
    folded = P.Propagator(fold(), lattice=lattice(8))(plane)
    leg = folded.legs[0]
    points = o.sampling.PointGrid(
        np.c_[np.zeros(5), np.zeros(5), np.linspace(-1, 1, 5)]
    )
    field = solution.field(points)
    coarse = lattice(11, 0.7)
    gauss = np.exp(-np.sum(coarse.points[..., :2] ** 2, -1) / 9)
    transverse = F.TransverseElectricField(coarse, 1.0, gauss[..., None] * [1, 0])
    spectrum = F.angular_spectrum(transverse, AIR)
    wave = F.PlaneBeam(
        F.profiles.Uniform(), F.polarization.x, wavelength0=1.0, region=AIR
    )
    waves = wave.plane_waves(0.0)
    ball = o.volumes.SphereVolume([0, 0, 0], 1.0)
    core = o.media.HomogeneousRegion("core", o.media.Dielectric(2.0, "core"))
    source, system = wave_problem()
    wave_propagator = o.alternatives.WavePropagator(system)
    interface = Y.PlanarInterface(Frame(), AIR, GLASS)
    zoo |= dict(
        frame=Frame([0, 0, 3e-3]),
        tilted_frame=Frame([1, 2, 3], TILTED),
        dielectric=o.media.Dielectric(1.5, "BK7-like"),
        air=o.media.air,
        vacuum=o.media.vacuum,
        region=GLASS,
        material_region=o.media.MaterialRegion("ball", ball, GLASS.material),
        distribution=o.media.MaterialDistribution(
            o.media.air, (o.media.MaterialRegion("ball", ball, GLASS.material),)
        ),
        plane=S.Plane.y(2.0),
        freeform_surface=S.FreeformSurface(np.sin, np.cos),
        axisymmetric=S.AxisymmetricSurface(np.square, np.square),
        cap=S.SphericalCap(-25e-3, Frame([0, 0, 3e-3])),
        sphere=S.Sphere(10.0),
        reversed_cap=S.SphericalCap(5.0).reversed(),
        reversed_oval=S.CartesianOval(-np.inf, 30, 1, 1.5).reversed(),
        reversed_ovoid=S.CartesianOvoid(-10, 30, 1, 1.5).reversed(),
        cylindrical_cap=S.CylindricalCap(5.0),
        cylinder=S.Cylinder(5.0, Frame([0, 0, -5])),
        asphere=thorlabs_al1225().elements[0].surface,
        oval=S.CartesianOval(-np.inf, 30, 1, 1.5),
        ovoid=S.CartesianOvoid(-10, 30, 1, 1.5),
        refraction=Y.Refraction(cap, AIR, GLASS, edge=6.25),
        reflection=Y.Reflection(cap, GLASS, behind=AIR),
        mirror=Y.Reflection(cap, AIR),
        stop=Y.Stop(4e-3, z=-0.5e-3),
        tilted_stop=Y.Stop(1.0, Frame([0, 0, 0], TILTED)),
        planar_interface=interface,
        plane_beam=plane,
        plane_wave=wave,
        focused_beam=focused,
        callable_beam=F.FocusedBeam(
            lambda x, y: np.clip(1 - (x**2 + y**2) / 0.0625, 0, None) ** 3,
            F.polarization.radial,
            focal_length=1.0,
            max_sine=0.25,
            wavelength0=1.0,
            region=o.media.Dielectric(1.33, "water"),
            apodization="none",
        ),
        carrier=focused.carrier,
        collimated=plane.carrier,
        diverging=F.Carrier.diverging([0, 0, 1]),
        custom_carrier=F.Carrier(np.sin, np.cos, np.tan),
        uniform=F.profiles.Uniform(),
        disk=F.profiles.Uniform(2.0),
        annulus=F.profiles.Annulus(1.0, 2.0),
        zones=F.profiles.Zones((1, 2, 3, 4, 5, 6, 7), (1, -1, 1, -1, 1, -1, 1, 0)),
        product=F.profiles.Gaussian(3.0) * F.profiles.Annulus(1.0, 2.0) * np.cos,
        gaussian=F.profiles.Gaussian(2e-3),
        jones=F.polarization.cartesian(1, 1j),
        radial=F.polarization.radial,
        electric=field.electric,
        electromagnetic=field,
        deferred=spectrum.propagate(1.0),
        transverse=transverse,
        plane_spectrum=spectrum,
        lattice_spectrum=folded.spectrum,
        hankel_spectrum=solution.spectrum,
        beam_spectrum=solution.spectra[0],
        expansion=waves,
        cartesian_grid=coarse,
        polar_grid=solution.spectrum.grid,
        point_grid=points,
        one_point=o.sampling.PointGrid([[0.0, 0.0, 1.0]]),
        surface_grid=cap.sample(u=np.linspace(-1, 1, 4), v=np.linspace(-1, 1, 4)),
        interface_map=leg.map,
        ray_map=leg.rays,
        refracted_rays=o.transport.RefractedRays(leg.map, (coarse.u, coarse.v)),
        lattice_transport=leg,
        meridian_transport=solution.legs[0],
        transport_report=solution.reports[0],
        propagator=meridian,
        solution=solution,
        lattice_solution=folded,
        wave_solution=wave_propagator(source),
        coverage=solution.coverage(points),
        plane_transport=P.PlaneTransport(),
        jet=o.observables.plane_wave_jet(waves, points),
        wave_propagator=wave_propagator,
        surface_transport=o.alternatives.SurfaceTransport(coarse, S.Plane(), AIR, 1.0),
        inverse_report=o.alternatives.InverseReport(1e-12, 3, 10.0, 1e-11, "lsqr"),
        spectral_fresnel=o.alternatives.SpectralFresnel(),
        local_fresnel=o.alternatives.LocalFresnel(),
        interface_solution=o.alternatives.scatter_plane(waves, interface),
        mie=o.references.MieSphere(waves, ball, GLASS),
        layered=o.references.LayeredSphere(
            waves, [ball, o.volumes.SphereVolume([0, 0, 0], 0.5)], [GLASS, core]
        ),
        infinite_cylinder=o.references.InfiniteCylinder(
            waves, o.volumes.CylinderVolume([0, 0, 0], [0, 1, 0], 1.0), GLASS
        ),
        sphere_volume=ball,
        cylinder_volume=o.volumes.CylinderVolume([0, 0, 0], [0, 2, 0], 1.0),
        between=o.volumes.VolumeBetweenSurfaces(
            S.Plane(), S.SphericalCap(-5.0, Frame([0, 0, 1.0])), 2.0
        ),
        bd=o.backend.bd,
    )
    return zoo


@pytest.fixture(scope="module")
def zoo():
    return build_zoo()


def public_classes():
    """Every concrete class of the public namespaces."""
    namespaces = [
        o.geometry,
        o.media,
        o.surfaces,
        o.system,
        o.fields,
        o.fields.profiles,
        o.fields.polarization,
        o.sampling,
        o.transport,
        o.propagation,
        o.observables,
        o.alternatives,
        o.references,
        o.volumes,
    ]
    found = set()
    for space in namespaces:
        names = getattr(space, "__all__", None) or dir(space)
        for name in names:
            value = getattr(space, name)
            if (
                inspect.isclass(value)
                and value.__module__.startswith("vecdiff")
                and not inspect.isabstract(value)
                and not name.startswith("_")
                and value is not o.sampling.Grid
                and not issubclass(value, Exception)
            ):
                found.add(value)
    return found


def test_the_zoo_holds_every_public_class(zoo):
    held = {type(value) for value in zoo.values()}
    missing = [
        c.__name__ for c in public_classes() if not any(issubclass(h, c) for h in held)
    ]
    assert missing == []


def test_every_repr_is_one_short_line_without_arrays_or_addresses(zoo):
    for name, value in zoo.items():
        text = repr(value)
        assert "\n" not in text, name
        assert len(text) <= 200, (name, text)
        assert "array(" not in text and "object at 0x" not in text, (name, text)
        assert "unprintable" not in text, (name, text)


def test_every_str_prints_without_arrays_or_addresses(zoo):
    for name, value in zoo.items():
        text = str(value)
        assert "array(" not in text and "object at 0x" not in text, (name, text)
        assert "unprintable" not in text, (name, text)


@pytest.mark.parametrize(
    "name, parts",
    [
        ("frame", ["Frame(origin=(0, 0, 0.003), normal=(0, 0, 1))"]),
        ("tilted_frame", ["origin=(1, 2, 3)", "normal=(0, -0.707107, 0.707107)"]),
        ("dielectric", ["Dielectric('BK7-like', n=1.5)"]),
        ("cap", ["SphericalCap(radius=-0.025, vertex=(0, 0, 0.003))"]),
        ("reversed_cap", ["radius=-5", "normal=(0, 0, -1)"]),
        ("asphere", ["curvature=78.2473", "conic=-0.6", "7 terms"]),
        ("oval", ["object_z=-inf", "image_z=30", "n_image=1.5"]),
        ("cylinder", ["Cylinder(radius=5", "axis=(0, 1, 0)"]),
        ("stop", ["Stop(radius=0.004, z=-0.0005)"]),
        ("refraction", ["'air' (n=1) -> 'glass' (n=1.5)", "edge=6.25"]),
        ("mirror", ["mirror in 'air'"]),
        ("gaussian", ["Gaussian(waist=0.002)"]),
        ("zones", ["edges=(1, 2, 3, ..., 5, 6, 7; 7 values)"]),
        ("product", ["Gaussian(waist=3)", "Annulus(inner=1, outer=2)", "cos"]),
        ("jones", ["circular", "(1, 1j)"]),
        (
            "plane_beam",
            ["PlaneBeam(Gaussian(waist=3), y-polarized", "origin=(0, 0, -5)"],
        ),
        ("focused_beam", ["focal_length=1", "max_sine=0.4", "focus=(0, 0, 15)"]),
        ("carrier", ["Carrier.converging(point=(0, 0, 15))"]),
        ("collimated", ["Carrier.collimated(direction=(0, 0, 1)"]),
        ("cartesian_grid", ["CartesianGrid(30 x 30, u in [-10.5, 9.8]", "plane z=0"]),
        ("point_grid", ["PointGrid(5 points"]),
        ("one_point", ["PointGrid(1 point at (0, 0, 1))"]),
        ("plane_spectrum", ["30 x 30 FFT modes", "in 'air'"]),
        ("hankel_spectrum", ["harmonics m = (-1, 1)", "in 'glass'", "forward"]),
        ("lattice_spectrum", ["lattice modes", "backward"]),
        ("transport_report", ["orders=(-1, 1)", "dark=0"]),
        ("mie", ["radius=1", "n 1.5 in 1", "x=6.28319", "order=30"]),
        ("layered", ["radii=(1, 0.5)", "n (1.5, 2) in 1"]),
        ("infinite_cylinder", ["axis=(0, 1, 0)", "x=6.28319"]),
        ("system lens", ["Stop", "Refraction(SphericalCap)", "'air' -> 'air')"]),
        ("system empty", ["OpticalSystem(no elements; in 'air')"]),
    ],
)
def test_a_repr_names_the_defining_parameters(zoo, name, parts):
    text = repr(zoo[name])
    for part in parts:
        assert part in text, (text, part)


def test_beams_print_sine_and_numerical_aperture_apart(zoo):
    text = str(zoo["callable_beam"])
    assert "sin(theta_max) = 0.25" in text
    assert "n sin(theta_max) = 0.3325" in text  # NA = 1.33 * 0.25
    assert "radial" in text and "water" in text


def test_fields_print_their_peaks_without_forming_a_deferred_h(zoo):
    deferred = zoo["deferred"]
    assert "formed on first read" in repr(deferred)
    assert "not formed yet" in str(deferred)
    assert "_form_magnetic" in deferred.__dict__
    text = str(zoo["electromagnetic"])
    assert "peak |E|" in text and "peak |H|" in text and "5 of 5" in text


# --- The system table -------------------------------------------------------


def body_rows(text):
    """The table rows of a printed system: the lines after the header's rule."""
    lines = text.splitlines()
    rule = next(i for i, line in enumerate(lines) if set(line) <= {"-", " "})
    return [line for line in lines[rule + 1 :] if line[:2].strip().isdigit()]


def test_the_lens_prints_one_row_per_element_with_its_prescription():
    text = str(lens())
    header = text.splitlines()[1].split()
    for column in ("element", "surface", "type", "z", "radius", "medium", "n"):
        assert column in header
    assert "aperture" in header and "thickness" in header
    rows = body_rows(text)
    assert len(rows) == 3
    stop, flat, curved = (row.split() for row in rows)
    assert stop[:3] == ["1", "Stop", "screen"] and "0.004" in stop
    assert "plane" not in rows[0]  # a stop's screen has no curvature
    assert flat[1:4] == ["Refraction", "S1", "Plane"] and "0.003" in flat
    assert "air -> BK7-like" in rows[1] and "1 -> 1.5" in rows[1]
    assert "-0.025" in curved and "1.5 -> 1" in rows[2]
    assert all(len(line) <= 110 for line in text.splitlines())
    assert "unit of wavelength0" in text
    assert "[2] BK7-like" in text


def test_table_returns_the_prescription_as_data():
    stop, flat, curved = lens().table()
    assert stop["element"] == "Stop" and stop["aperture"] == 4e-3
    assert stop["thickness"] == pytest.approx(0.5e-3)
    assert flat["form"] == "plane" and flat["radius"] is None
    assert flat["thickness"] == pytest.approx(3e-3)
    assert (flat["n_before"], flat["n_after"]) == (1.0, 1.5)
    assert curved["radius"] == -25e-3 and curved["vertex"] == (0.0, 0.0, 3e-3)
    assert curved["thickness"] is None


def test_a_ghost_marks_the_surfaces_it_meets_again():
    system = ghost()
    rows = [row.split() for row in body_rows(str(system))]
    assert [row[2] for row in rows] == ["S1", "S2", "S1*", "S2*"]
    data = system.table()
    assert [row["repeated"] for row in data] == [False, False, True, True]
    assert data[2]["first_met"] == 1
    assert "element 3 meets S1 again" in str(system)
    assert "glass, behind air " in str(system)  # the material, not 'air 2'
    assert "Regions of the segments: [0] air, [1] glass" in str(system)
    assert "Regions of the segments" not in str(cassegrain())
    assert [row["thickness"] for row in data[:3]] == [100.0, -100.0, 100.0]


def test_a_tilted_mirror_shows_its_tilt_and_normal():
    text = str(fold())
    assert "coaxial" not in text.splitlines()[0]
    assert "45°" in text and "(0, -0.707107, 0.707107)" in text
    assert "Mirror" in text


def test_the_cassegrain_prints_its_conics_and_negative_thickness():
    rows = cassegrain().table()
    assert [r["conic"] for r in rows] == [None, -1.0, -1.96]
    assert rows[1]["radius"] == pytest.approx(-100.0)
    assert rows[1]["thickness"] == pytest.approx(-40.0)
    assert "k -1.96" in str(cassegrain())


def test_a_freeform_has_no_radius_and_an_asphere_its_terms():
    text = str(systems()["freeform"])
    assert "freeform" in text and "reversed CartesianOval" in text
    assert "A16 = -4.96041e26" in str(thorlabs_al1225())


class Cells(HTMLParser):
    """The text of every th and td of an HTML table, row by row."""

    def __init__(self):
        super().__init__()
        self.rows, self.depth, self.cell = [], 0, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        if tag in ("td", "th"):
            self.cell = ""

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self.rows[-1].append(self.cell)
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data


def test_the_html_table_holds_the_same_cells():
    system = ghost()
    html = system._repr_html_()
    parser = Cells()
    parser.feed(html)
    parser.close()
    header, *rows = parser.rows
    assert header[:4] == ["#", "element", "surface", "type"]
    assert len(rows) == 4 and all(len(r) == len(header) for r in rows)
    assert [r[2] for r in rows] == ["S1", "S2", "S1*", "S2*"]
    assert rows[1][header.index("medium")] == "glass, behind air"
    assert html.count("<table>") == html.count("</table>") == 1
    assert "| S1* |" in system._repr_markdown_()


def test_every_system_prints():
    for name, system in systems().items():
        text = str(system)
        assert text.startswith("OpticalSystem"), name
        assert len(body_rows(text)) == len(system) if len(system) else True
        assert system._repr_html_().startswith("<div>")


# --- The Solution and the formatting ---------------------------------------


def test_a_solution_prints_one_report_row_per_element_and_flags_near_refusal(zoo):
    text = str(zoo["solution"])
    rows = body_rows(text)
    assert len(rows) == 1 and "meridian" in rows[0] and "(-1, 1)" in rows[0]
    assert "Leaves as HankelSpectrum in 'glass'" in text
    near = o.transport.TransportReport(0.0, 0, 0.0, folded_relative_amplitude=0.004)
    solution = zoo["solution"]
    flagged = P.Solution(solution.spectra, solution.elements, (near,))
    assert "0.4!" in str(flagged)
    assert "Solution" in flagged._repr_html_()


def test_numbers_are_compact_and_columns_align_on_the_decimal_point():
    assert [number(x) for x in (0.003, 5.32e-7, -25e-3, 1.0, 0.1 + 0.2)] == [
        "0.003",
        "5.32e-7",
        "-0.025",
        "1",
        "0.3",
    ]
    assert number(1 - 2j) == "1-2j" and number(np.float64(-0.0)) == "0"
    table = text_table(["z"], [["-0.5"], ["12.25"]], "n").splitlines()
    assert table[2].index(".") == table[3].index(".")


def test_printing_never_raises():
    class Broken:
        @guarded
        def __repr__(self):
            raise RuntimeError("no")

        @guarded
        def _repr_html_(self):
            raise RuntimeError("no")

    assert repr(Broken()) == "<Broken (unprintable: RuntimeError)>"
    assert Broken()._repr_html_() is None
