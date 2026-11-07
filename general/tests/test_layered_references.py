"""Layered sphere and infinite cylinder: exact references and their Debye terms.

Prototype values come from the archived validation code (mie_shell.py,
cyl_exact.py), converted to the library normalization of the beam.
"""

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose
from vecdiff_general.observables.diagnostics import interface_residuals
from vecdiff_general.propagation.references import (
    InfiniteCylinder,
    LayeredSphere,
    MieSphere,
)
from vecdiff_general.propagation.references.cylinder import _focused_density
from vecdiff_general.propagation.references.diagnostics import maxwell_residuals

host = o.backend.to_numpy
Z0 = o.media.medium.Z0


def region(n, name="r"):
    return o.media.HomogeneousRegion(name, o.media.Dielectric(n))


AIR, GLASS, DENSE = region(1.0, "air"), region(1.5, "glass"), region(2.0, "dense")
CIRCULAR = o.fields.polarization.cartesian(1 / np.sqrt(2), 1j / np.sqrt(2))


def wave(direction, polarization, region=AIR):
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    p = np.asarray(polarization, complex)
    p = p - d * (d @ p)
    k = 2 * np.pi * region.material.n
    return o.fields.PlaneWaveExpansion([k * d], [p], [0.1, -0.2, 0.3], 1.0, region)


def converging(max_sine, focus, profile=None):
    """The manuscript's Appendix C beam (circular input) converging to focus."""
    return o.fields.FocusedBeam(
        profile or o.fields.profiles.Uniform(max_sine),
        CIRCULAR,
        focal_length=1.0,
        max_sine=max_sine,
        wavelength0=1.0,
        region=AIR,
        frame=o.geometry.Frame(focus),
        apodization="none",
    )


def spheres(*radii, centre=(0, 0, 0)):
    return [o.volumes.SphereVolume(centre, r) for r in radii]


def grid(points):
    return o.sampling.PointGrid(np.asarray(points, float))


def test_one_layer_is_mie_inside_outside_and_term_by_term():
    source = wave([0.2, -0.3, 1], [1, 0.4j, 0])
    sphere = o.volumes.SphereVolume([0.2, 0.1, -0.3], 2.0)
    mie, layered = (
        MieSphere(source, sphere, GLASS),
        LayeredSphere(source, [sphere], [GLASS]),
    )
    assert_allclose(layered.a, mie.a, rtol=0, atol=1e-13)
    assert_allclose(layered.b, mie.b, rtol=0, atol=1e-13)
    rng = np.random.default_rng(1)
    points = grid(host(sphere.center) + rng.normal(size=(60, 3)) * 1.6)
    for term in (None, 1, 2):
        expected = mie.evaluate(points, term=term)
        actual = layered.evaluate(points, term=term)
        scale = np.abs(expected.electric.values).max()
        assert_allclose(
            actual.electric.values, expected.electric.values, atol=1e-12 * scale
        )
        assert_allclose(actual.magnetic, expected.magnetic, atol=1e-12 * scale / Z0)
    q = layered.efficiencies()
    assert_allclose(q["scattering"], q["extinction"], rtol=1e-12)
    assert_allclose(q["scattering"], mie.efficiencies()["scattering"], rtol=1e-12)


def test_index_matched_layers_reduce_to_the_homogeneous_sphere():
    beam = converging(0.6, [0.3, -0.2, 3.0])
    rng = np.random.default_rng(2)
    points = grid(rng.uniform(-3.2, 3.2, size=(80, 3)))
    # Shell index equal to the exterior: Mie of the core.
    hidden = LayeredSphere(beam, spheres(2.5, 1.5), [AIR, GLASS])
    expected = MieSphere(beam, spheres(1.5)[0], GLASS)
    # Core index equal to the shell: Mie of the outer sphere.
    filled = LayeredSphere(beam, spheres(2.5, 1.5), [GLASS, GLASS])
    outer = MieSphere(beam, spheres(2.5)[0], GLASS)
    for layered, mie in ((hidden, expected), (filled, outer)):
        a, b = layered.evaluate(points), mie.evaluate(points)
        scale = np.abs(b.electric.values).max()
        assert_allclose(a.electric.values, b.electric.values, atol=2e-12 * scale)
        assert_allclose(a.magnetic, b.magnetic, atol=2e-12 * scale / Z0)
    # An index-matched interface is transparent term by term: first entries agree.
    inside = grid(rng.uniform(-0.8, 0.8, size=(20, 3)))
    assert_allclose(
        hidden.evaluate(inside, term=1).electric.values,
        expected.evaluate(inside, term=1, side="interior").electric.values,
        atol=1e-12,
    )


def test_debye_terms_sum_to_the_total_in_every_region():
    beam = converging(0.5, [0.2, 0.0, 4.0])
    layered = LayeredSphere(beam, spheres(3.0, 2.2, 1.0), [GLASS, DENSE, GLASS])
    rng = np.random.default_rng(3)
    for j, (low, high) in enumerate(((3.2, 4.0), (2.3, 2.9), (1.1, 2.1), (0.0, 0.9))):
        direction = rng.normal(size=(12, 3))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        points = grid(direction * rng.uniform(low, high, 12)[:, None])
        total = layered.evaluate(points).electric.values
        if j == 0:
            total = total - beam.evaluate(points).electric.values
        first = 0 if j == 0 else 1
        terms = sum(
            layered.evaluate(points, term=t).electric.values
            for t in (*range(first, 4), ("after", 3))
        )
        assert_allclose(terms, total, atol=1e-11 * np.abs(total).max())


def test_layered_fields_satisfy_maxwell_and_both_interface_conditions():
    source = wave([0.3, 0.1, 1], [0.2, 1, 0])
    layered = LayeredSphere(source, spheres(2.0, 1.2), [GLASS, DENSE])
    theta = np.linspace(0.1, 3.0, 9)
    phi = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    T, P = np.meshgrid(theta, phi)
    normal = np.stack(
        (np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)), -1
    ).reshape(-1, 3)
    for j, radius in enumerate(layered.radii):
        surface = grid(radius * normal)
        residual = interface_residuals(
            layered.evaluate(surface, side=j + 1),
            layered.evaluate(surface, side=j),
            normal,
            layered.regions[j + 1].material,
            layered.regions[j].material,
            reference_electric=1.0,
            reference_index=1.0,
        )
        assert max(residual.values()) < 1e-11
    residual = maxwell_residuals(
        layered.evaluate, [[0.3, 0.9, 1.0]], wavelength0=1.0, index=1.5, step=0.005
    )
    assert max(residual.values()) < 1e-5
    with pytest.raises(ValueError, match="region 2"):
        layered.evaluate(grid([[0, 0, 1.5]]), side="interior")


def test_tunnelling_core_stays_exact_at_large_size():
    # Air core in glass: orders between k0 R2 and k0 R1 tunnel into the core;
    # its coefficients are ~exp(tau) and must not overflow or lose the field.
    source = wave([0, 0, 1], [1, 0, 0])
    layered = LayeredSphere(source, spheres(200.0, 170.0), [GLASS, AIR])
    theta = np.linspace(0.05, 3.1, 7)
    normal = np.stack((np.sin(theta), 0 * theta, np.cos(theta)), -1)
    surface = grid(170.0 * normal)
    residual = interface_residuals(
        layered.evaluate(surface, side=2),
        layered.evaluate(surface, side=1),
        normal,
        AIR.material,
        GLASS.material,
        reference_electric=1.0,
        reference_index=1.0,
    )
    assert max(residual.values()) < 1e-9
    q = layered.efficiencies()
    assert_allclose(q["scattering"], q["extinction"], rtol=1e-10)


@pytest.mark.parametrize(
    "geometry, values",
    [
        (
            # n 1 | 1.5 | 2, R2 = R1 / 2, beam towards 1.5 R1, u_max 0.4
            (1.5, 2.0, 0.5, 1.5, 0.4),
            {
                "first": [
                    [
                        0.0083547757108 - 0.970718677394j,
                        0.970718677394 + 0.0083547757108j,
                        0,
                    ],
                    [
                        -0.0029079200469 + 0.0100037856691j,
                        0.0147585003943 - 0.0053440292649j,
                        0.0692043877152 - 0.0351740594911j,
                    ],
                    [
                        -0.0041845037212 - 0.0021118401817j,
                        0.0062770276261 - 0.0042033336342j,
                        0.0085028329669 - 0.0015084938723j,
                    ],
                ],
                "full": [
                    [
                        -0.0103069664196 - 0.9685991068303j,
                        0.9685991068303 - 0.0103069664196j,
                        0,
                    ],
                    [
                        -0.0089073420045 + 0.0085677675628j,
                        0.016280446164 - 0.0164471177976j,
                        0.0687882661937 - 0.0365446418131j,
                    ],
                    [
                        0.000562351122 - 0.0003859038862j,
                        0.0021848798954 + 0.0012247554928j,
                        0.0099275170665 - 0.0019097618248j,
                    ],
                ],
            },
        ),
        (
            # meniscus in air: n 1 | 1.5 | 1, R2 = 0.85 R1, beam towards 0.75 R1, u_max 0.65
            (1.5, 1.0, 0.85, 0.75, 0.65),
            {
                "first": [
                    [
                        0.7796528970402 + 0.2098961578655j,
                        -0.2098961578655 + 0.7796528970402j,
                        0,
                    ],
                    [
                        0.2727359584948 + 0.0691323709153j,
                        -0.0626618849436 + 0.2455034397894j,
                        0.0496010694856 - 0.1776665142602j,
                    ],
                    [
                        0.0379639569429 + 0.013119443947j,
                        -0.0154189860165 + 0.0437215176356j,
                        0.0057411340211 - 0.0147376819062j,
                    ],
                ],
            },
        ),
    ],
    ids=["aplanatic_pair", "meniscus"],
)
def test_two_surface_fields_match_the_archived_reference(geometry, values):
    """R1 = 30 lambda0, hard pupil, image plane of the prototype (shell_cases.py).

    The archived total field of the meniscus sets the orders that tunnel into
    its air core to zero (their closed form overflows there), so only its
    first entry is quoted; those orders change its total by 6e-9.
    """
    n1, n2, r2, focus, max_sine = geometry
    R1 = 30.0
    image = {0.5: 11.25, 0.85: 23.538461538486114}[r2]
    layered = LayeredSphere(
        converging(max_sine, [0, 0, focus * R1]),
        spheres(R1, r2 * R1),
        [region(n1), region(n2)],
    )
    points = grid([[r, 0, image] for r in (0.0, 0.7, 2.3)])
    for key in values:
        term = 1 if key == "first" else None
        actual = layered.evaluate(points, term=term).electric.values
        assert_allclose(actual, np.array(values[key]), rtol=0, atol=2e-11)


def test_cylinder_index_matched_returns_the_incident_field():
    cylinder = o.volumes.CylinderVolume([0.1, 0, -0.2], [0.2, 1, 0.1], 1.7)
    rng = np.random.default_rng(4)
    points = grid(rng.uniform(-3, 3, size=(50, 3)))
    for incident in (
        wave([0.4, 0.5, 0.8], [1, 0.3j, 0.2]),
        converging(0.5, [0.2, 0.1, 2.0]),
    ):
        same = InfiniteCylinder(incident, cylinder, AIR, reach=4.0).evaluate(points)
        expected = incident.evaluate(points)
        scale = np.abs(expected.electric.values).max()
        assert_allclose(
            same.electric.values, expected.electric.values, atol=1e-10 * scale
        )
        assert_allclose(same.magnetic, expected.magnetic, atol=1e-10 * scale / Z0)


@pytest.mark.parametrize("polarization", ["parallel", "perpendicular"])
def test_cylinder_normal_incidence_matches_bohren_huffman(polarization):
    """Scattered axial field: -b_n (E along the axis) or -a_n (H along it), BH 8.4."""
    from scipy.special import h1vp, hankel1, jv, jvp

    radius, m = 2.0, 1.5
    cylinder = o.volumes.CylinderVolume([0, 0, 0], [0, 1, 0], radius)
    E = [0, 1, 0] if polarization == "parallel" else [1, 0, 0]
    source = o.fields.PlaneWaveExpansion([[0, 0, 2 * np.pi]], [E], [0, 0, 0], 1.0, AIR)
    reference = InfiniteCylinder(source, cylinder, GLASS)
    rng = np.random.default_rng(5)
    points = grid(
        np.c_[rng.uniform(-4, 4, 25), rng.uniform(-1, 1, 25), rng.uniform(2.2, 4, 25)]
    )
    total, incident = reference.evaluate(points), source.evaluate(points)
    if polarization == "parallel":
        scattered = (total.electric.values - incident.electric.values)[:, 1]
    else:
        scattered = (total.magnetic - incident.magnetic)[:, 1] * Z0
    x, N = 2 * np.pi * radius, reference.order
    n = np.arange(-N, N + 1)[:, None]
    J, dJ, Jm, dJm = jv(n, x), jvp(n, x), jv(n, m * x), jvp(n, m * x)
    H, dH = hankel1(n, x), h1vp(n, x)
    if polarization == "parallel":
        coefficient = (Jm * dJ - m * dJm * J) / (Jm * dH - m * dJm * H)
    else:
        coefficient = (m * Jm * dJ - dJm * J) / (m * Jm * dH - dJm * H)
    local = host(points.points) @ reference.basis
    rho, phi = np.hypot(local[:, 0], local[:, 1]), np.arctan2(local[:, 1], local[:, 0])
    u, v, _ = reference.basis.T
    alpha = np.arctan2(v[2], u[2])  # direction of incidence, +z
    series = -np.sum(
        1j ** (n % 4)
        * np.exp(-1j * n * alpha)
        * coefficient
        * hankel1(n, 2 * np.pi * rho[None])
        * np.exp(1j * n * phi[None]),
        0,
    )
    assert_allclose(scattered, series, atol=1e-11 * np.abs(series).max())


def test_cylinder_oblique_incidence_debye_continuity_maxwell_and_energy():
    cylinder = o.volumes.CylinderVolume([0, 0, 0], [0, 1, 0], 2.0)
    source = wave([0.6, 0.45, 0.66], [0.3, 1, 0.5j])
    reference = InfiniteCylinder(source, cylinder, GLASS)
    rng = np.random.default_rng(6)
    inside = grid(
        np.c_[
            rng.uniform(-1.3, 1.3, 30),
            rng.uniform(-2, 2, 30),
            rng.uniform(-1.3, 1.3, 30),
        ]
    )
    outside = grid(
        np.c_[rng.uniform(2.1, 4, 30), rng.uniform(-2, 2, 30), rng.uniform(-4, 4, 30)]
    )
    total = reference.evaluate(inside).electric.values
    terms = sum(
        reference.evaluate(inside, term=t).electric.values for t in (1, 2, ("after", 2))
    )
    assert_allclose(terms, total, atol=1e-12 * np.abs(total).max())
    total = (
        reference.evaluate(outside).electric.values
        - source.evaluate(outside).electric.values
    )
    terms = sum(
        reference.evaluate(outside, term=t).electric.values
        for t in (0, 1, ("after", 1))
    )
    assert_allclose(terms, total, atol=1e-12 * np.abs(total).max())
    phi, y = rng.uniform(0, 2 * np.pi, 20), rng.uniform(-2, 2, 20)
    normal = np.c_[np.sin(phi), 0 * y, np.cos(phi)]
    surface = grid(np.c_[2.0 * normal[:, 0], y, 2.0 * normal[:, 2]])
    residual = interface_residuals(
        reference.evaluate(surface, side="interior"),
        reference.evaluate(surface, side="exterior"),
        normal,
        GLASS.material,
        AIR.material,
        reference_electric=1.0,
        reference_index=1.0,
    )
    assert max(residual.values()) < 1e-10
    residual = maxwell_residuals(
        reference.evaluate, [[0.4, 0.3, -0.5]], wavelength0=1.0, index=1.5, step=0.005
    )
    assert max(residual.values()) < 1e-5
    # Lossless: every (h, n) mode returns the incoming power, axial E weighted by n².
    h = np.array([0.0, 2.1, -4.0])
    incident = rng.normal(size=(2, 2 * reference.order + 1, 3)) + 1j * rng.normal(
        size=(2, 2 * reference.order + 1, 3)
    )
    mantissa, scale = reference._response(h, incident, None)["exterior"]
    outgoing = incident / 2 + mantissa * np.exp(scale)
    weight = np.array([1.0, 1.0])[:, None, None]  # n_o = 1
    assert_allclose(
        np.sum(weight * np.abs(outgoing) ** 2, 0),
        np.sum(weight * np.abs(incident / 2) ** 2, 0),
        rtol=1e-10,
    )


def test_cylinder_first_entry_matches_the_archived_reference_and_guards_reach():
    """R = 20 lambda0, smooth pupil, beam towards 1.5 R, u_max 0.25 (cyl_exact.py)."""
    radius = 20.0
    beam = converging(
        0.25,
        [0, 0, 1.5 * radius],
        lambda x, y: np.clip(1 - (x**2 + y**2) / 0.25**2, 0, None) ** 3,
    )
    theta, phi, _, density = beam.quadrature(10.0)
    s, _, _ = beam._directions(theta, phi)
    assert_allclose(
        _focused_density(beam, s), density, atol=1e-15 * np.abs(density).max()
    )
    cylinder = o.volumes.CylinderVolume([0, 0, 0], [0, 1, 0], radius)
    reference = InfiniteCylinder(beam, cylinder, GLASS, reach=2.0)
    xs, ys = (0.0, 0.9, -2.1), (0.0, 1.3)
    points = grid([[[x, y, radius / 1.5] for x in xs] for y in ys])
    expected = [
        [
            [0.0151578456126 - 0.0306595154642j, 0.0308194154798 + 0.015303095422j, 0],
            [
                0.0067672525147 - 0.0183910593077j,
                0.0183915254477 + 0.0067471155189j,
                -0.0024342336032 - 0.0013961849183j,
            ],
            [
                0.0005624288265 + 0.0001810062682j,
                -0.0002656974334 + 0.0005693289678j,
                0.0003363646569 - 4.17594932e-05j,
            ],
        ],
        [
            [
                0.0100511665926 - 0.0287794000441j,
                0.0289341226922 + 0.0101759474843j,
                0.0007743954429 - 0.0003275460228j,
            ],
            [
                0.0036904327999 - 0.0170270707144j,
                0.0170974531612 + 0.0036554901842j,
                -0.0018502267474 - 0.0012454556227j,
            ],
            [
                0.0003146620153 + 0.0001143309698j,
                -0.0002007386845 + 0.0003412701815j,
                0.0003137696384 - 0.0001042121338j,
            ],
        ],
    ]
    actual = reference.evaluate(points, term=1).electric.values
    # The archived k_y quadrature (800 nodes) is converged to about 1e-10 relative.
    assert_allclose(actual, np.array(expected), rtol=0, atol=5e-12)
    with pytest.raises(ValueError, match="axial reach"):
        reference.evaluate(grid([[0, 3.0, 13.0]]))
    with pytest.raises(TypeError, match="CylinderVolume"):
        InfiniteCylinder(beam, spheres(1.0)[0], GLASS)


def test_sphere_debye_coefficients_keep_relative_accuracy_beyond_the_size():
    """One-way coefficients at size 2000 for n = 1990..2090 against 40-digit mpmath.

    A direct solve in the Hankel basis lost these orders (T21 by 1e-9 here and
    by 1e4 at size 126; the exterior term 0 entirely).
    """
    from vecdiff_general.propagation.references.debye import sphere_debye_coefficients

    EXACT = {
        "TE": {
            "T21": [
                -0.39223940441963245 + 0.5609569737402873j,
                0.006315340734626892 + 0.026367321809188438j,
                -1.1847739974543229e-05 + 5.51866388144495e-05j,
                -1.5106048519555116e-08 + 7.919960601111846e-09j,
            ],
            "R11": [
                -0.6190312167024795 - 0.5517705153137744j,
                0.8916154214581135 - 0.45225198812101974j,
                0.9118820769898615 + 0.4104522816853384j,
                -0.5687742050771397 + 0.8224937103947164j,
            ],
            "T12": [
                -0.26149293627975495 + 0.3739713158268582j,
                0.0042102271564179285 + 0.01757821453945896j,
                -7.898493316362152e-06 + 3.679109254296633e-05j,
                -1.0070699013036743e-08 + 5.279973734074564e-09j,
            ],
            "zero": [
                -0.8653648043891398 - 0.19601312945599j,
                -0.00012267835842374073 - 0.00037873972614209996j,
                -5.30989009200935e-10 - 1.108529039039429e-09j,
                -4.84864129663862e-17 - 7.890046437443509e-17j,
            ],
            "mie": [
                -0.1112646202608044 + 0.3843962390455554j,
                0.05830041163463324 + 7.793905397691746e-06j,
                -0.00013445323910836414 + 4.81593999674089e-13j,
                -9.629205063843028e-09 + 1.0045325314111562e-24j,
            ],
        },
        "TM": {
            "T21": [
                -0.5190192970148584 + 0.7684984476498078j,
                0.01558338105807223 + 0.035713503343378354j,
                2.6614154052979014e-06 + 7.748170493744804e-05j,
                -1.610660010231067e-08 + 1.568347451596776e-08j,
            ],
            "R11": [
                -0.5071238359155125 - 0.41171772519170025j,
                0.6802253949349573 - 0.7323122385258506j,
                0.9976430778336183 - 0.06861694574714536j,
                -0.02661526564689467 + 0.9996457510711206j,
            ],
            "T12": [
                -0.3460128646765722 + 0.5123322984332052j,
                0.010388920705381485 + 0.023809002228918904j,
                1.7742769368652677e-06 + 5.1654469958298694e-05j,
                -1.0737733401540448e-08 + 1.0455649677311839e-08j,
            ],
            "zero": [
                -0.7856681959647955 - 0.15831992724193675j,
                -0.0002532106329690129 - 0.0003117204430267179j,
                -1.0017496235234018e-09 - 7.241178195116451e-10j,
                -8.423232329146071e-17 - 3.9423550783786154e-17j,
            ],
            "mie": [
                -0.19083759347177306 + 0.5620438396162094j,
                0.048762412328833854 + 1.3109280296490812e-05j,
                0.0011291919804661567 + 3.211398146739564e-11j,
                -1.5689032338868953e-08 + 1.90532415433879e-24j,
            ],
        },
    }
    actual = sphere_debye_coefficients(
        2000.0, 1.5, 1.0, np.array([1990, 2030, 2060, 2090])
    )
    for polarization, values in EXACT.items():
        for key, expected in values.items():
            assert_allclose(actual[polarization][key], expected, rtol=2e-11)


def test_references_sum_the_incident_field_by_the_chosen_method():
    pytest.importorskip("finufft")
    beam = converging(0.5, [0.2, 0.0, 3.0])
    rng = np.random.default_rng(8)
    direction = rng.normal(size=(40, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    outside = grid(direction * rng.uniform(1.6, 2.4, 40)[:, None])
    for reference in (
        MieSphere(beam, spheres(1.5)[0], GLASS),
        LayeredSphere(beam, spheres(1.5, 1.0), [GLASS, DENSE]),
        InfiniteCylinder(
            beam, o.volumes.CylinderVolume([0, 0, 0], [0, 1, 0], 1.5), GLASS, reach=3.0
        ),
    ):
        direct = reference.evaluate(outside).electric.values
        fast = reference.evaluate(outside, method="nufft").electric.values
        assert_allclose(fast, direct, atol=1e-10 * np.abs(direct).max())
