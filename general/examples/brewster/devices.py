"""Brewster devices of revolution: closed forms and surface sequences.

At Brewster incidence theta_B = arctan(n) the incidence and refraction angles
add to 90 degrees, so every ray is turned by the same angle

    delta = theta_B - theta_t = 2 theta_B - 90 deg,  sin delta = (n² - 1)/(n² + 1),

whatever its height, and the ratio of the s and p amplitude transmissions is

    t_s / t_p = cos(theta_i - theta_t) = cos(delta) = 2 n / (1 + n²).

On a surface of revolution met by a beam along its axis, p is the radial
field and s the azimuthal one. A sequence of Brewster surfaces is therefore a
constant rotation of the meridional ray field by a sum of +-delta, with
E_phi / E_rho multiplied by cos(delta) at each surface and T_p = 1.

* Alternating shell (``shell``): two parallel cones of slope n; the rotations
  cancel, the beam stays collimated and is shifted radially by
  d sin(delta) / cos(theta_t) (d the wall thickness normal to the wall);
  spike (A, inward shift) and funnel (B, outward shift) shells alternate.
* Double-Brewster axicon (``brewster_axicon``): an entrance cone of slope n
  and an exit surface of slope tan(3 theta_B - 180 deg), met at the internal
  Brewster angle so that the two rotations add: the output is a converging
  conical wave of half-angle 2 delta, NA = sin(2 delta) = 4 n (n² - 1)/(n² + 1)²,
  with T_p = 1 at both faces. For n = sqrt(3) the exit face is a plane
  (NA 0.866); for n = 2, NA = 0.96; for n = 1 + sqrt(2) (theta_B = 67.5 deg,
  diamond) delta = 45 deg and the output grazes the axis (NA 1).
"""

import numpy as np

import meridional_operator as mo


def brewster(n):
    """theta_B, theta_t, delta (radians), cos(delta) = t_s/t_p, and NA = sin(2 delta)."""
    theta_b = np.arctan(n)
    theta_t = np.pi / 2 - theta_b
    delta = theta_b - theta_t
    return dict(
        theta_b=theta_b,
        theta_t=theta_t,
        delta=delta,
        amplitude_ratio=np.cos(delta),
        power_ratio=np.cos(delta) ** 2,
        na=np.sin(2 * delta),
        exit_slope=np.tan(3 * theta_b - np.pi),
    )


def shell(z, kind, n, separation, tip=0.0):
    """Two parallel cones of slope +-n, ``separation`` apart on the axis.

    kind "A": apex upstream (spike), rays tilt inward in the glass;
    kind "B": apex downstream (funnel), rays tilt outward.
    """
    g = n if kind == "A" else -n
    return [
        mo.cone(z, g, 1.0, n, tip, f"shell {kind} entrance"),
        mo.cone(z + separation, g, n, 1.0, tip, f"shell {kind} exit"),
    ]


def shell_stack(kinds, n, separation, aperture, gap, z0=0.0, tip=0.0):
    """Shells in the order of ``kinds`` ("ABAB..."); a funnel after a spike is
    set back by 2 n ``aperture`` so the two do not meet within that radius."""
    surfaces, z, previous = [], z0, None
    for kind in kinds:
        if previous == "A" and kind == "B":
            z += 2 * n * aperture
        if previous is not None:
            z += gap
        surfaces += shell(z, kind, n, separation, tip)
        z += separation
        previous = kind
    return surfaces, z


def brewster_axicon(z, n, depth, tip=0.0):
    """Entrance cone of slope n at z, exit surface at z + depth met at internal
    Brewster incidence by the refracted rays; returns the surfaces and the
    input radii (inner, outer) whose rays leave through the exit before
    crossing the axis and where the two faces have not met."""
    b = brewster(n)
    slope = b["exit_slope"]
    delta = b["delta"]
    exit_face = (
        mo.plane(z + depth, n, 1.0, "axicon exit")
        if abs(slope) < 1e-12
        else mo.cone(z + depth, slope, n, 1.0, 0.0, "axicon exit")
    )
    surfaces = [mo.cone(z, n, 1.0, n, tip, "axicon entrance"), exit_face]
    # A ray entering at rho reaches the exit at
    # rho (1 + (n - g) tan d / (1 + g tan d)) - depth tan d / (1 + g tan d).
    tan = np.tan(delta)
    inner = depth * tan / (1 + slope * tan + (n - slope) * tan)
    outer = depth / (n - slope)
    return surfaces, (inner, outer)
