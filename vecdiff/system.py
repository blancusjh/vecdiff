"""Aperture limits of a two-surface element.

Three separate things can stop a ray, and which one binds decides what a design
can actually deliver:

**The faces meet.**  A lens is bounded by its two surfaces.  Going outwards from
the vertex the first face advances and the second retreats, so at some radius
they cross and the element has run out of glass.  That radius is set by the
axial separation and, through the sags, by the conjugates -- which is why it
moves when ``z0`` moves even though nothing about the aperture was touched.

**Grazing incidence.**  On a surface with ``n0 < ni`` the incidence angle grows
with radius and reaches ninety degrees at a finite radius.  Past it the closed
form continues onto a sheet the object no longer illuminates.

**The critical angle.**  On a surface with ``n0 > ni`` the *transmitted* angle
reaches ninety degrees first, at ``sin(theta_0) = ni / n0``.  A stigmatic
surface cannot be continued past it: beyond that radius Snell has no real
solution, so no oval exists that would send the ray to the image point.

The last one is why a dense-to-rare exit surface is such a hard ceiling.  A
single fused-silica-to-air stigmatic surface saturates near ``NA = 0.74`` no
matter how the conjugates are chosen; getting past that needs either a
higher-index final element or an immersion medium, which is exactly the route
real high-aperture lithography took.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class ApertureBudget:
    """Which constraint binds, and at what radius."""

    radius: float
    limited_by: str
    intersection_radius: float
    surfaces_intersect: bool
    first_limit: float
    second_limit: float

    def __str__(self) -> str:
        return (
            f"r = {self.radius:.4f} limited by {self.limited_by} "
            f"(faces meet at {self.intersection_radius:.4f}"
            f"{'' if self.surfaces_intersect else ', outside the domain'}; "
            f"first surface {self.first_limit:.4f}, second {self.second_limit:.4f})"
        )


def surface_intersection_radius(first, second, separation) -> tuple[float, bool]:
    """Return the radius where two facing surfaces meet, and whether they do.

    ``separation`` is the axial distance between the two vertices.  The
    surfaces meet where ``sag1(r) - sag2(r) = separation``; both sags are
    measured from their own vertex, so a positive ``sag1`` and a negative
    ``sag2`` both eat into the axial gap.

    When the surfaces do not meet anywhere the closed forms are still real, the
    second return value is ``False`` and the radius is the end of the common
    domain.
    """
    separation = float(separation)
    ceiling = min(first.r_fold, second.r_fold) * (1.0 - 1e-9)

    def gap(r):
        s1 = float(first.sag(np.array([r]))[0])
        s2 = float(second.sag(np.array([r]))[0])
        return (s1 - s2) - separation

    if gap(ceiling) < 0.0:
        return float(ceiling), False
    if gap(1e-12) > 0.0:
        return 0.0, True
    return float(brentq(gap, 1e-12, ceiling, xtol=1e-13)), True


def element_aperture(first, second, separation) -> ApertureBudget:
    """Return the usable aperture radius of a two-surface element."""
    r_edge, meets = surface_intersection_radius(first, second, separation)
    lim1 = first.aperture_limit
    lim2 = second.aperture_limit

    radius = min(r_edge, lim1)
    if radius == r_edge and r_edge < lim1:
        which = "the faces meeting"
    else:
        which = _refraction_label(first, "first")

    return ApertureBudget(
        radius=float(radius),
        limited_by=which,
        intersection_radius=float(r_edge),
        surfaces_intersect=meets,
        first_limit=float(lim1),
        second_limit=float(lim2),
    )


def _refraction_label(surface, position: str) -> str:
    if surface.n0 < surface.ni:
        return f"grazing incidence on the {position} surface"
    return f"the critical angle on the {position} surface"


def relay_throughput(first, second, separation, *, n_probe: int = 4000):
    """Trace the chain and report where it is actually stopped.

    The ray leaving ``first`` at meridional angle ``alpha_i`` reaches the second
    vertex plane at radius ``|z0_2| tan(alpha_i)``, so the second surface's own
    aperture limit reaches back and caps the first surface's usable radius.
    Returns ``None`` when nothing gets through at all.
    """
    budget = element_aperture(first, second, separation)
    if budget.radius <= 0.0:
        return None

    r1 = np.linspace(1e-9, budget.radius * (1.0 - 1e-4), int(n_probe))
    g1 = first.ray_geometry(r1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = abs(second.z0) * g1.sin_ai / np.maximum(np.abs(g1.cos_ai), 1e-15)

    admitted = (r2 <= second.aperture_limit) & g1.valid
    if not np.any(admitted):
        return None

    last = int(np.flatnonzero(admitted)[-1])
    g2 = second.ray_geometry(np.array([r2[last] * (1.0 - 1e-4)]))

    stopped_by = (
        budget.limited_by
        if last >= n_probe - 2
        else _refraction_label(second, "second")
    )

    na_image = second.ni * float(g2.sin_ai[0])
    na_object = first.n0 * float(g1.sin_a0[last])
    return dict(
        na_image=na_image,
        na_object=na_object,
        reduction=na_image / na_object if na_object > 0 else np.inf,
        radius_first=float(r1[last]),
        radius_second=float(r2[last]),
        budget=budget,
        limited_by=stopped_by,
    )
