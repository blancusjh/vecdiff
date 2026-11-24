"""Every public entry point given an absorbing medium works or says why not.

What is exact with a complex index works: the Fresnel reflection of an
absorbing medium behind a surface, and plane waves in an absorbing region.
Everything else (transport along rays through an absorbing medium, beams
whose quadratures take a real wavenumber, the Bessel-kernel references, the
wave-transport alternatives, energy and momentum densities) raises
media.AbsorbingMediumError, early and with its reason, never a raw
TypeError from float() of a complex number.
"""

import numpy as np
import pytest
import vecdiff as o

F, S, P, Y, M = o.fields, o.surfaces, o.propagation, o.system, o.media
Frame = o.geometry.Frame
METAL = M.Medium(0.05 + 3.3j, "silver")
LOSSY = M.Medium(1.5 + 0.02j, "lossy glass")
GLASS = M.Dielectric(1.5, "glass")
INSIDE = M.HomogeneousRegion("inside", LOSSY)


def lattice(count=32, step=0.3):
    axis = (np.arange(count) - count // 2) * step
    return o.sampling.CartesianGrid(axis, axis)


def sampled(region=None):
    grid = lattice()
    values = np.zeros(grid.shape + (2,), complex)
    values[..., 0] = np.exp(-(grid.points[..., 0] ** 2 + grid.points[..., 1] ** 2) / 4)
    return F.TransverseElectricField(grid, 1.0, values)


def beam(region=M.air, z=-10.0):
    return F.PlaneBeam(
        F.profiles.Gaussian(3.0),
        F.polarization.x,
        wavelength0=1.0,
        region=region,
        frame=Frame([0, 0, z]),
    )


def plane(z=0.0):
    return S.Plane(Frame([0, 0, z]))


def absorbing_spectrum():
    return F.angular_spectrum(sampled(), INSIDE)


WORKS = {
    "a metal mirror, meridian": lambda: P.Propagator(
        Y.Reflection(plane(), M.air, behind=METAL)
    )(beam()),
    "a metal mirror, lattice": lambda: P.Propagator(
        Y.Reflection(plane(), M.air, behind=METAL), lattice=lattice(96, 0.4)
    )(beam()),
    "a metal mirror, sampled field": lambda: P.Propagator(
        Y.Reflection(plane(), M.air, behind=METAL)
    )(sampled(), F.Carrier.collimated([0, 0, 1.0])),
    "a lossy glass behind a curved mirror": lambda: P.Propagator(
        Y.Reflection(S.SphericalCap(-80.0), M.air, behind=LOSSY), fans=((15.0, 25.0),)
    )(beam()),
    "an internal reflection off an absorbing ghost": lambda: P.Propagator(
        Y.OpticalSystem(
            Y.Refraction(plane(), M.air, GLASS),
            Y.Reflection(plane(2.0), GLASS, behind=METAL),
        ),
        lattice=lattice(96, 0.4),
    )(beam()),
    "the angular spectrum": absorbing_spectrum,
    "its propagation": lambda: absorbing_spectrum().propagate(1.0),
    "a plane transport": lambda: P.PlaneTransport()(absorbing_spectrum(), 0.5),
    "propagate": lambda: P.propagate(sampled(), 0.5, INSIDE),
    "a plane-wave expansion": lambda: F.PlaneWaveExpansion(
        [[0, 0, 2 * np.pi * LOSSY.n]], [[1.0, 0, 0]], np.zeros(3), 1.0, INSIDE
    ),
    "the propagator in an absorbing region": lambda: P.Propagator(
        Y.OpticalSystem(medium=INSIDE)
    )(sampled()).field(z=1.0, shape=(8, 8)),
    "the Poynting vector there": lambda: o.observables.poynting(
        absorbing_spectrum().propagate(0.5)
    ),
}


@pytest.mark.parametrize("name", list(WORKS))
def test_what_is_exact_with_an_absorbing_medium_works(name):
    WORKS[name]()


def element_leg(encounter):
    return lambda: o.transport.LatticeTransport(
        lattice(), encounter, F.Carrier.collimated([0, 0, 1.0]), 1.0
    )


REFUSED = {
    # Transport along rays through an absorbing medium: at construction.
    "a refraction into a metal": (
        lambda: P.Propagator(Y.Refraction(plane(), M.air, METAL)),
        "carrying light across element 1",
    ),
    "a refraction out of a lossy glass": (
        lambda: P.Propagator(Y.Refraction(plane(), LOSSY, M.air)),
        "carrying light across element 1",
    ),
    "a mirror inside a lossy glass": (
        lambda: P.Propagator(Y.Reflection(plane(), LOSSY)),
        "carrying light across element 1",
    ),
    "a reflection inside a lossy glass": (
        lambda: P.Propagator(Y.Reflection(plane(), LOSSY, behind=M.air)),
        "carrying light across element 1",
    ),
    "a second element beyond a metal": (
        lambda: P.Propagator(
            Y.OpticalSystem(
                Y.Refraction(plane(), M.air, GLASS),
                Y.Refraction(plane(1.0), GLASS, METAL),
            )
        ),
        "carrying light across element 2",
    ),
    "a stop in a lossy glass": (
        lambda: P.Propagator(Y.OpticalSystem(Y.Stop(2.0), medium=INSIDE)),
        "carrying light across element 1",
    ),
    "a stop in the light's absorbing region": (
        lambda: P.Propagator(Y.Stop(2.0))(
            absorbing_spectrum(), F.Carrier.collimated([0, 0, 1])
        ),
        "carrying light across element 1",
    ),
    "a lattice leg into a metal": (
        element_leg(Y.Refraction(plane(), M.air, METAL)),
        "carrying light along rays",
    ),
    "a meridian leg into a metal": (
        lambda: o.transport.MeridianTransport(
            Y.Refraction(plane(), M.air, METAL),
            F.Carrier.collimated([0, 0, 1.0]),
            Frame(),
            1.0,
            scale=5.0,
        ),
        "carrying light along rays",
    ),
    "a ray map into a metal": (
        lambda: o.transport.InterfaceMap(
            plane(), F.Carrier.collimated([0, 0, 1.0]), 1.0, METAL.n
        ),
        "ray map's the transmitted medium absorbs",
    ),
    "a ray map from a lossy glass": (
        lambda: o.transport.InterfaceMap(
            plane(), F.Carrier.collimated([0, 0, 1.0]), LOSSY.n, 1.0, branch="reflected"
        ),
        "ray map's the incident medium absorbs",
    ),
    # Beams and the spectra whose quadratures take a real wavenumber.
    "a plane beam in a lossy glass": (lambda: beam(LOSSY), "a beam"),
    "a focused beam in a metal": (
        lambda: F.FocusedBeam(
            F.profiles.Uniform(1.0),
            F.polarization.x,
            focal_length=1.0,
            max_sine=0.5,
            wavelength0=1.0,
            region=METAL,
        ),
        "a beam",
    ),
    "a beam spectrum relabelled into a lossy glass": (
        lambda: F.BeamSpectrum(beam(), lattice(), INSIDE),
        "a beam",
    ),
    "a Hankel spectrum": (
        lambda: F.HankelSpectrum(
            Frame(), INSIDE, 1.0, np.array([0.0, 1.0]), np.ones(2), {}
        ),
        "a Hankel spectrum",
    ),
    "a lattice spectrum": (
        lambda: F.LatticeSpectrum(lattice(), INSIDE, 1.0, np.zeros((32, 32, 2))),
        "a lattice spectrum",
    ),
    "the angular spectrum's propagating disk": (
        lambda: F.angular_spectrum(sampled(), INSIDE, evanescent="discard"),
        "no propagating disk",
    ),
    "the paraxial model": (
        lambda: absorbing_spectrum().propagate(1.0, model="fresnel"),
        "paraxial",
    ),
    # Energy and momentum densities.
    "the energy density": (
        lambda: o.observables.energy_density(
            absorbing_spectrum().propagate(0.5), LOSSY
        ),
        "energy density",
    ),
    "the spin density": (
        lambda: o.observables.spin_density(absorbing_spectrum().propagate(0.5), LOSSY),
        "spin density",
    ),
    # The wave-transport alternatives.
    "the wave propagator": (
        lambda: o.alternatives.WavePropagator(Y.Refraction(plane(), M.air, LOSSY)),
        "wave propagator",
    ),
    "a surface transport": (
        lambda: o.alternatives.SurfaceTransport(lattice(), plane(), INSIDE, 1.0),
        "surface transport",
    ),
}


@pytest.mark.parametrize("name", list(REFUSED))
def test_what_is_not_exact_with_an_absorbing_medium_says_why(name):
    make, what = REFUSED[name]
    with pytest.raises(M.AbsorbingMediumError, match=what) as refusal:
        make()
    message = str(refusal.value)
    assert "absorb" in message and "ABSORBING_MEDIA.md" in message


def test_the_references_refuse_an_absorbing_index_naming_their_kernels():
    from vecdiff.references import InfiniteCylinder, LayeredSphere, MieSphere
    from vecdiff.references.debye import (
        cylinder_single_interface,
        sphere_debye_coefficients,
        sphere_single_interface,
    )

    light = beam()
    ball = o.volumes.SphereVolume([0, 0, 0], 5.0)
    rod = o.volumes.CylinderVolume([0, 0, 0], [0, 1.0, 0], 5.0)
    makers = [
        lambda: MieSphere(light, ball, INSIDE),
        lambda: LayeredSphere(light, (ball,), (INSIDE,)),
        lambda: InfiniteCylinder(light, rod, INSIDE),
        lambda: sphere_single_interface(np.ones((1, 3)), 1.0, 5.0, 1.0, LOSSY.n),
        lambda: cylinder_single_interface(
            np.ones((1, 3)), 1.0, 5.0, 1.0, LOSSY.n, "TE"
        ),
        lambda: sphere_debye_coefficients(10.0, METAL.n, 1.0, np.arange(1, 4)),
    ]
    for make in makers:
        with pytest.raises(M.AbsorbingMediumError, match="real arguments only"):
            make()


def test_the_references_take_a_material_or_its_region():
    from vecdiff.references import LayeredSphere, MieSphere

    light = beam()
    ball, core = (o.volumes.SphereVolume([0, 0, 0], r) for r in (5.0, 2.0))
    sphere = MieSphere(light, ball, GLASS)
    assert sphere.interior == M.HomogeneousRegion("glass", GLASS)
    layered = LayeredSphere(light, (ball, core), (GLASS, M.Dielectric(1.8, "core")))
    assert [r.material.n for r in layered.interiors] == [1.5, 1.8]
    with pytest.raises(TypeError, match="media.Medium"):
        MieSphere(light, ball, 1.5)


def test_decaying_waves_on_a_plane_are_summed_by_the_transform_as_directly():
    """The decay exp(-Im k_z z) folds into the amplitudes on a plane parallel
    to the spectrum's, so FINUFFT sums them; off such a plane the exact
    direct sum is taken."""
    spectrum = absorbing_spectrum()
    x, y = np.meshgrid(np.linspace(-3, 3, 9), np.linspace(-2, 2, 7))
    on_plane = o.sampling.PointGrid(np.stack((x, y, np.full_like(x, 0.8)), -1))
    fast = spectrum.evaluate(on_plane, method="nufft")
    exact = spectrum.evaluate(on_plane, method="direct")
    for got, want in (
        (fast.electric.values, exact.electric.values),
        (fast.magnetic, exact.magnetic),
    ):
        assert np.abs(got - want).max() < 1e-10 * np.abs(want).max()
    scattered = o.sampling.PointGrid(np.c_[x.ravel(), y.ravel(), 0.1 * x.ravel()])
    fast = spectrum.electric(scattered, method="nufft").values
    exact = spectrum.electric(scattered, method="direct").values
    assert np.array_equal(fast, exact)
