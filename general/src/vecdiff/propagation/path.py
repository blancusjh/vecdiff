"""The path as the light meets it: every surface seen along the propagation.

A leg carries light across a surface it meets along the surface's normal,
from below its graph. Light that arrives against a surface's normal (after
a reflection) meets the surface's reverse (``GraphSurface.reversed``,
frame u, -v, -n); light inside a closed surface that leaves it travelling
along its normal meets its far side (``ImplicitSurface.far_side``). Which
view each element is met in follows from the light's axis ray, traced
through the path by the laws of refraction and reflection: the view whose
normal the ray travels along and which the ray enters first.
"""

from dataclasses import dataclass, replace

import numpy as np

from vecdiff._backend import bd, to_numpy
from vecdiff.geometry.frames import Frame
from vecdiff.surfaces.surface import ImplicitSurface
from vecdiff.system import Stop


@dataclass(frozen=True)
class OrientedPath:
    """The elements of a path, each with its surface seen as the light meets it.

    ``elements`` are the system's elements with their surfaces replaced by
    those views, ``axes`` the frame of each leg's propagation axis (the
    light's axis frame, reversed where the light travels against it),
    ``points`` and ``directions`` where the axis ray meets each element and
    the way it travels on from there.
    """

    elements: tuple
    axes: tuple
    points: tuple
    directions: tuple


def views(surface):
    """The views a surface can be met in along a normal.

    The surface and its reverse; for a closed surface, also its far side and
    that reversed, where it declares one.
    """
    out = [surface, surface.reversed()]
    if isinstance(surface, ImplicitSurface):
        try:
            far = surface.far_side()
        except NotImplementedError:
            return out
        out += [far, far.reversed()]
    return out


class _Line:
    """A carrier stand-in: the axis ray is traced by the map's own intersection."""

    def directions(self, points):  # pragma: no cover - never evaluated
        raise NotImplementedError

    jacobians = directions


def entry(view, point, direction):
    """Signed distance along the line to where it enters ``view`` from below,
    and the unit normal there (global); NaN where it does not."""
    from vecdiff.transport.rays import InterfaceMap

    frame = view.frame
    p = bd.asarray((point - frame.origin) @ frame.basis)[None]
    s = bd.asarray(direction @ frame.basis)[None]
    if float(s[0, 2]) <= 1e-12:
        return np.nan, None
    if isinstance(view, ImplicitSurface):
        leg = float(to_numpy(view.crossing(p, s))[0])
        if not np.isfinite(leg):
            return np.nan, None
        _, gradient, _ = view.implicit(p + leg * s)
        normal = to_numpy(gradient)[0]
    else:
        leg = float(to_numpy(InterfaceMap(view, _Line(), 1.0, 1.0)._intersect(p, s))[0])
        if not np.isfinite(leg):
            return np.nan, None
        q = to_numpy(p + leg * s)[0]
        if not bool(to_numpy(view.domain(bd.asarray(q[None, :2])))[0]):
            return np.nan, None
        slope = to_numpy(view.gradient(bd.asarray(q[None, :2])))[0]
        normal = np.r_[-slope, 1.0]
    normal = normal / np.linalg.norm(normal)
    return leg, to_numpy(frame.basis) @ normal


def onward(element, direction, normal):
    """The axis ray's direction after the encounter, by the law of the element."""
    cosine = float(direction @ normal)
    if isinstance(element, Stop):
        return direction
    if element.branch != "transmitted":
        return direction - 2 * cosine * normal
    eta = element.incident.material.n / element.far.material.n
    if isinstance(eta, complex):
        # Between absorbing media only an index-matched boundary is carried
        # (the identity); its ratio is one.
        eta = eta.real
    tangent = eta * (direction - cosine * normal)
    radicand = 1 - float(tangent @ tangent)
    if radicand <= 0:
        return None
    return tangent + np.sqrt(radicand) * normal


def view_of(element, surface):
    """The element with its surface replaced by a view of it."""
    if isinstance(element, Stop):
        return replace(element, frame=surface.frame)
    return replace(element, surface=surface)


def axis_frame(axis, normal):
    """The light's axis frame, reversed where the light travels against it."""
    along = float(to_numpy(axis.normal) @ to_numpy(normal))
    return axis if along > 0 else axis.reversed()


def orient(elements, axis, direction, wavelength):
    """The OrientedPath of ``elements`` for light along the line of ``axis``.

    ``axis`` is the light's frame (its origin on the axis ray) and
    ``direction`` the way the ray travels from it. The first element is met
    where the line from far upstream first enters one of its views; each
    later one where the ray, from the previous encounter, first enters one
    of its views beyond it.
    """
    origin = to_numpy(axis.origin).astype(float)
    direction = np.asarray(to_numpy(direction), float)
    direction = direction / np.linalg.norm(direction)
    reach = max(
        [
            float(np.linalg.norm(to_numpy(e.surface.frame.origin) - origin))
            for e in elements
        ]
        + [1e3 * wavelength]
    )
    point = origin - 4 * reach * direction
    oriented, axes, points, directions = [], [], [], []
    previous = None
    for index, element in enumerate(elements, 1):
        surface = element.surface
        tolerance = 1e-9 * reach if surface == previous else -1e-9 * reach
        best = None
        for view in views(surface):
            leg, normal = entry(view, point, direction)
            if np.isfinite(leg) and leg > tolerance and (best is None or leg < best[0]):
                best = (leg, normal, view)
        if best is None:
            raise ValueError(
                f"the light's axis ray does not meet element {index} "
                f"({type(element).__name__} on a {type(surface).__name__}) after "
                "the one before it: the path is out of order or misses it"
            )
        leg, normal, view = best
        point = point + leg * direction
        oriented.append(view_of(element, view))
        axes.append(axis_frame(Frame(axis.origin, axis.basis), view.frame.normal))
        following = onward(element, direction, normal)
        if following is None:
            raise ValueError(
                f"the light's axis ray is totally reflected at element {index}; "
                "a reflection there is a system.Reflection"
            )
        direction = following / np.linalg.norm(following)
        points.append(point)
        directions.append(direction)
        previous = surface
    return OrientedPath(tuple(oriented), tuple(axes), tuple(points), tuple(directions))
