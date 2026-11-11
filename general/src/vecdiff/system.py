"""The optical system: the ordered path of the light's encounters.

An ``OpticalSystem`` is the path light follows, element after element. A
``Refraction`` carries it across a surface into the medium on the other
side, a ``Reflection`` turns it back into its own medium, and a ``Stop``
lets through what passes the hole of an opaque screen. A surface may be
met more than once: a ghost path, or an internal reflection, is another
system built from the same surfaces.

A medium (``media.Medium``, a ``media.Dielectric`` when lossless) is what
fills a part of space; a region (``media.HomogeneousRegion``) is a part of
space, with its medium. The light
between two encounters is one segment of the path, and it lies in one
region; several segments can lie in the same region (in front of a mirror
the incident and the reflected light, inside a plate the light reflected
back and forth). The system derives the regions from the path: at a
Reflection or a Stop the light stays in its region, and at a Refraction it
crosses into the region on the surface's other side, the one already met
there if the surface was met before. Elements name media; a region may be
named instead wherever its identity matters.

``PlanarInterface`` is the planar boundary of the plane-wave scattering
alternative (``alternatives``), not an element of a path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from vecdiff._backend import bd
from vecdiff.geometry.frames import Frame
from vecdiff.media.medium import HomogeneousRegion, Medium


@dataclass(frozen=True)
class PlanarInterface:
    """Frame normal points from the minus region into the plus region."""

    frame: Frame
    minus: HomogeneousRegion
    plus: HomogeneousRegion

    def __post_init__(self):
        if not isinstance(self.frame, Frame):
            raise TypeError("interface requires a Frame")

        if not all(isinstance(r, HomogeneousRegion) for r in (self.minus, self.plus)):
            raise TypeError("interface sides must be homogeneous regions")

        if self.minus == self.plus:
            raise ValueError(
                "name the two spatial regions distinctly, even for equal materials"
            )


def region_of(medium):
    """The region a medium names: a region is itself, a material alone is the
    region named after it."""
    if medium is None or isinstance(medium, HomogeneousRegion):
        return medium
    return HomogeneousRegion(medium.name, medium)


def material_of(medium):
    """The material of a medium or of the region it fills."""
    return medium.material if isinstance(medium, HomogeneousRegion) else medium


def check_medium(value, what, optional=False):
    if (value is None and optional) or isinstance(value, (Medium, HomogeneousRegion)):
        return value
    raise TypeError(
        f"{what} must be a media.Medium (or Dielectric) or a media.HomogeneousRegion"
    )


def check_surface(surface):
    from vecdiff.surfaces.surface import GraphSurface

    if not isinstance(surface, GraphSurface):
        raise TypeError("an encounter needs a placed surface (a surfaces.GraphSurface)")
    return surface


def check_edge(edge):
    if edge is not None and (not bd.isfinite(edge) or edge <= 0):
        raise ValueError("an edge is a positive radius, or None for no edge")
    return None if edge is None else float(edge)


@dataclass(frozen=True)
class Refraction:
    """Light crossing a surface from one medium into another.

    ``surface`` is a placed surface; ``before`` is the medium the light
    arrives through and ``after`` the medium it enters (each a
    media.Medium, or the media.HomogeneousRegion it fills). The light
    may meet the surface along its normal or against it.

    ``edge`` is the radius of the face in the surface's chart, where the
    material ends. It bounds the encounter, as the chart's own domain does:
    rays that meet the surface beyond it miss the face and leave this path
    (their share is reported), so the path's field ends there. The light
    that passes outside the face belongs to another path; the edge itself
    scatters nothing. A screen that absorbs is a Stop.
    """

    surface: object
    before: Medium | HomogeneousRegion
    after: Medium | HomogeneousRegion
    edge: float | None = None

    branch = "transmitted"

    def __post_init__(self):
        check_surface(self.surface)
        check_medium(self.before, "before")
        check_medium(self.after, "after")
        object.__setattr__(self, "edge", check_edge(self.edge))

    @property
    def incident(self):
        """The region the light arrives through."""
        return region_of(self.before)

    @property
    def outgoing(self):
        """The region the light leaves into."""
        return region_of(self.after)

    @property
    def far(self):
        """The region on the surface's other side."""
        return region_of(self.after)

    @property
    def aperture(self):
        """The radius beyond which rays miss the encounter: the edge, or None."""
        return self.edge


@dataclass(frozen=True)
class Reflection:
    """Light turned back by a surface into the medium it arrived through.

    With ``behind``, the medium on the surface's other side, it is the
    Fresnel reflection of that boundary: a ghost, an internal reflection,
    total internal reflection, and the reflection of a metal or a lossy
    glass (a ``behind`` of complex index). Without it the surface is an
    ideal mirror, on which tangential E vanishes. ``edge`` bounds the face
    as for a Refraction.
    """

    surface: object
    medium: Medium | HomogeneousRegion
    behind: Medium | HomogeneousRegion | None = None
    edge: float | None = None

    def __post_init__(self):
        check_surface(self.surface)
        check_medium(self.medium, "medium")
        check_medium(self.behind, "behind", optional=True)
        object.__setattr__(self, "edge", check_edge(self.edge))

    @property
    def branch(self):
        return "mirror" if self.behind is None else "reflected"

    @property
    def incident(self):
        return region_of(self.medium)

    @property
    def outgoing(self):
        return region_of(self.medium)

    @property
    def far(self):
        return region_of(self.behind)

    @property
    def aperture(self):
        return self.edge


@dataclass(frozen=True, init=False)
class Stop:
    """An opaque screen across the path, with a circular hole.

    The screen is the plane z = 0 of ``frame`` (or, with ``z``, the plane
    z = const of the global axes), and the hole of ``radius`` is centred on
    the frame's origin. A stop is an element of the path placed in space,
    not part of a surface: it lies in the medium of its segment
    (``medium``, which the system supplies), absorbs what meets the screen
    outside the hole and lets through, unchanged, what passes it. The field
    beyond is the cut field, and it diffracts from the cut.
    """

    radius: float
    frame: Frame
    medium: Medium | HomogeneousRegion | None

    branch = "transmitted"

    def __init__(self, radius, frame=None, *, z=None, medium=None):
        if frame is not None and z is not None:
            raise ValueError("place a stop by its frame or by z, not both")
        if frame is None:
            frame = Frame([0.0, 0.0, 0.0 if z is None else float(z)])
        if not isinstance(frame, Frame):
            raise TypeError("a stop is placed by a geometry.Frame")
        if not bd.isfinite(radius) or radius <= 0:
            raise ValueError("a stop's hole has a positive radius")
        check_medium(medium, "medium", optional=True)
        object.__setattr__(self, "radius", float(radius))
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "medium", medium)

    @property
    def surface(self):
        """The plane of the screen."""
        from vecdiff.surfaces.surface import Plane

        return Plane(self.frame)

    @property
    def incident(self):
        return region_of(self.medium)

    outgoing = far = incident

    @property
    def aperture(self):
        """The radius of the hole: rays that meet the screen beyond it are absorbed."""
        return self.radius


ELEMENTS = (Refraction, Reflection, Stop)


class _Regions:
    """The regions met along a path: one per part of space, each surface's two sides."""

    def __init__(self):
        self.regions, self.sides = [], []

    def name(self, material):
        """A region for ``material`` named after it, distinct from those met."""
        names = {region.name for region in self.regions}
        name, count = material.name, 1
        while name in names:
            count += 1
            name = f"{material.name} {count}"
        return HomogeneousRegion(name, material)

    def known(self, region):
        if region is not None and region not in self.regions:
            self.regions.append(region)
        return region

    def across(self, surface, here, given, index, what):
        """The region on ``surface``'s other side from ``here``: the one met
        there before, else ``given`` (a region, or a material to name one)."""
        other = next(
            (
                pair[1] if pair[0] == here else pair[0]
                for s, pair in self.sides
                if s == surface and here in pair
            ),
            False,
        )
        if other is False:
            other = (
                given
                if given is None or isinstance(given, HomogeneousRegion)
                else self.name(given)
            )
            self.sides.append((surface, (here, other)))
        elif given is not None and not agrees(other, given):
            raise ValueError(
                f"element {index}: {what} of this surface is {describe(other)}, "
                f"met there before, not {describe(given)}"
            )
        return self.known(other)


def agrees(region, medium):
    """Whether a region is ``medium``: that region, or one of that material."""
    if region is None:
        return medium is None
    if isinstance(medium, HomogeneousRegion):
        return region == medium
    return region.material == medium


def describe(medium):
    if medium is None:
        return "nothing (an ideal mirror)"
    if isinstance(medium, HomogeneousRegion):
        return f"region {medium.name!r} ({medium.material.name}, n = {medium.material.n:g})"
    return f"{medium.name} (n = {medium.n:g})"


def arriving(element):
    """The medium an element says the light arrives through, or None."""
    if isinstance(element, Refraction):
        return element.before
    return element.medium


@dataclass(frozen=True, init=False)
class OpticalSystem:
    """The ordered path of the light's encounters, and the regions it crosses.

    ``OpticalSystem(*elements)`` (or one sequence of them) takes Refraction,
    Reflection and Stop elements in the order the light meets them, and
    ``medium`` the medium the light arrives in when no element names it
    (a system of stops alone, or none). Adjacent elements must agree on the
    medium between them; after a Reflection or a Stop the light is in the
    medium it arrived in. ``elements`` holds them with every medium
    resolved to its region and ``regions`` the region of each segment of
    the path, the light arriving first and leaving last.
    """

    elements: tuple
    regions: tuple

    def __init__(self, *elements, medium=None):
        if len(elements) == 1 and not isinstance(elements[0], ELEMENTS):
            elements = tuple(elements[0])
        for index, element in enumerate(elements, 1):
            if not isinstance(element, ELEMENTS):
                raise TypeError(
                    f"element {index} is a {type(element).__name__}: a path holds "
                    "system.Refraction, system.Reflection and system.Stop"
                )
        check_medium(medium, "medium", optional=True)
        resolved, regions = resolve(elements, medium)
        object.__setattr__(self, "elements", resolved)
        object.__setattr__(self, "regions", regions)

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    @property
    def region(self):
        """The region the light arrives in."""
        return self.regions[0]


def resolve(elements, medium):
    """The elements with their media resolved to regions, and each segment's region."""
    first = next((arriving(e) for e in elements if arriving(e) is not None), None)
    if first is None:
        first = medium
    elif medium is not None and not (
        material_of(first) == material_of(medium)
        and not (
            isinstance(medium, HomogeneousRegion)
            and isinstance(first, HomogeneousRegion)
            and medium != first
        )
    ):
        raise ValueError(
            f"medium= is {describe(medium)}, but the first element says the light "
            f"arrives through {describe(first)}"
        )
    book = _Regions()
    here = None
    if first is not None:
        here = book.known(
            first if isinstance(first, HomogeneousRegion) else book.name(first)
        )
    regions, resolved = [here], []
    for index, element in enumerate(elements, 1):
        said = arriving(element)
        if said is not None and not agrees(here, said):
            raise ValueError(
                f"element {index} ({type(element).__name__}) says the light arrives "
                f"through {describe(said)}, but it is in {describe(here)}"
            )
        if isinstance(element, Stop):
            element = replace(element, medium=here)
        elif isinstance(element, Reflection):
            behind = book.across(
                element.surface, here, element.behind, index, "the medium behind"
            )
            if behind is None and element.behind is not None:
                raise ValueError(f"element {index}: this surface is a mirror")
            element = replace(element, medium=here, behind=behind)
        else:
            after = book.across(
                element.surface, here, element.after, index, "the medium beyond"
            )
            if after is None:
                raise ValueError(
                    f"element {index}: light cannot cross a surface met before as "
                    "an ideal mirror"
                )
            element = replace(element, before=here, after=after)
            here = after
        resolved.append(element)
        regions.append(here)
    return tuple(resolved), tuple(regions)
