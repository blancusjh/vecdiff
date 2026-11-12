"""The refusal of an absorbing medium where a computation is exact only in a
transparent one, and the reasons given for it.

docs/ABSORBING_MEDIA.md derives what is exact with a complex index (Fresnel
reflection from an absorbing medium behind a surface; plane waves in an
absorbing region) and analyses what the rest would require.
"""

from vecdiff.media.medium import HomogeneousRegion, Medium

NOTE = "docs/ABSORBING_MEDIA.md"

# Why each kind of computation needs a transparent medium.
TRANSPORT = (
    "a transport leg carries light along real rays through the medium it "
    "arrives in and the one it enters, and in an absorbing medium the "
    "transmitted wave is inhomogeneous and its rays and eikonal complex. "
    "Only a reflection from an absorbing medium behind the surface "
    "(system.Reflection(..., behind=<absorbing>)), whose rays stay in the "
    "transparent medium, is carried"
)
BEAM = (
    "a beam's quadratures over its plane waves take a real wavenumber; give "
    "the field on a plane (fields.ElectricField or TransverseElectricField) "
    "and expand it with fields.angular_spectrum, which is exact there"
)
REFERENCE = (
    "this exact solution's Bessel and Riccati-Bessel kernels "
    "(references._bessel) take real arguments only"
)
ALTERNATIVE = (
    "the wave-transport alternatives carry plane waves of real wavevector "
    "across real-index boundaries only"
)
SPECTRUM = (
    "it holds the propagating plane waves of a transparent region, sampled in "
    "their real polar angle; a transport leg's output, which is transparent "
    "where a leg is supported"
)
OBSERVABLE = (
    "the energy and momentum densities of absorbing (and so dispersive) "
    "matter are not determined by its index at one frequency"
)


class AbsorbingMediumError(ValueError):
    """An absorbing medium given to a computation that is exact only in a
    transparent one."""


def region_of(medium, what):
    """The region a medium names: a region is itself, a Medium alone the
    region named after it; anything else is refused naming ``what``."""
    if isinstance(medium, HomogeneousRegion):
        return medium
    if isinstance(medium, Medium):
        return HomogeneousRegion(medium.name, medium)
    raise TypeError(f"{what} must be a media.Medium or a media.HomogeneousRegion")


def material(medium):
    """The material of a medium or of the region it fills."""
    return medium.material if isinstance(medium, HomogeneousRegion) else medium


def describe(medium):
    m = material(medium)
    if isinstance(medium, HomogeneousRegion) and medium.name != m.name:
        return f"region {medium.name!r} ({m.name}, n = {m.n:g})"
    return f"{m.name} (n = {m.n:g})"


def require_real(n, what, why):
    """The index ``n`` unchanged, unless it is complex with Im n != 0: then
    AbsorbingMediumError naming ``what``, and ``why``."""
    value = complex(n)
    if value.imag != 0:
        raise AbsorbingMediumError(
            f"{what} is not supported for an absorbing medium (n = {value:g}): "
            f"{why} (see {NOTE})"
        )
    return n


def require_transparent(medium, what, why):
    """``medium`` (a Medium or a region; None passes), unless it absorbs:
    then AbsorbingMediumError naming ``what`` is refused, and ``why``."""
    m = material(medium)
    if isinstance(m, Medium) and m.absorbing:
        raise AbsorbingMediumError(
            f"{what} is not supported in an absorbing medium, "
            f"{describe(medium)}: {why} (see {NOTE})"
        )
    return medium
