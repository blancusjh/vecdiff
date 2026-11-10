"""Momentum and angular momentum of monochromatic light in a homogeneous medium.

Densities follow the dual-symmetric definitions of Berry (2009) and Bliokh,
Bekshaev & Nori (2013) for SI phasors with time dependence exp(-i omega t),
permittivity eps and permeability mu = mu0:

    w   = (eps |E|² + mu |H|²) / 4                               energy
    S   = Im(eps E* x E + mu H* x H) / (4 omega)                 spin
    h   = -sqrt(eps mu) Im(E* . H) / (2 omega)                   helicity
    p_o = Im(eps E* . (grad) E + mu H* . (grad) H) / (4 omega)   canonical momentum
    p_s = curl S / 2                                             spin momentum
    L   = (r - origin) x p_o                                     orbital angular momentum

and eps mu P = p_o + p_s for the Poynting vector P. A plane wave of helicity
sigma carries S = sigma w / omega along its direction and p_o = n w / c.
``part="electric"`` or ``"magnetic"`` keeps one half, doubled, as in the
electric-biased definitions; for a single plane wave the parts agree. The
momenta need gradients and take an ElectricFieldJet (E, its gradient and
Hessian), from which grad H follows exactly. Units: J s/m³ for S, h and L,
N s/m³ for p_o and p_s.

Force and torque integrate the time-averaged Maxwell stress tensor
T = Re(eps E E* + mu H H*) / 2 - w I over a closed, outward-oriented surface
quadrature enclosing the body and lying in the homogeneous medium.
"""

from vecdiff_general._backend import bd, to_numpy
from vecdiff_general.media.medium import C0, MU0, Z0

_PARTS = {"dual": (0.5, 0.5), "electric": (1.0, 0.0), "magnetic": (0.0, 1.0)}


def _part(part):
    if part not in _PARTS:
        raise ValueError('part must be "dual", "electric" or "magnetic"')

    return _PARTS[part]


def _omega(field):
    return 2 * bd.pi * C0 / field.electric.wavelength0


def _magnetic_gradient(jet):
    """d_i H_l from the Hessian of E: H = curl E / (i k0 Z0) in a nonmagnetic medium."""
    h = jet.hessian
    curl = bd.stack(
        (
            h[..., 1, 2] - h[..., 2, 1],
            h[..., 2, 0] - h[..., 0, 2],
            h[..., 0, 1] - h[..., 1, 0],
        ),
        axis=-1,
    )

    return curl / (1j * 2 * bd.pi / jet.electric.wavelength0 * Z0)


def spin_density(field, material, *, part="dual"):
    """Spin angular momentum density S of an ElectromagneticField."""
    electric, magnetic = _part(part)
    e, h = field.electric.values, field.magnetic
    spin = electric * material.epsilon * bd.imag(bd.cross(e.conj(), e))
    spin = spin + magnetic * MU0 * bd.imag(bd.cross(h.conj(), h))

    return spin / (2 * _omega(field))


def helicity_density(field, material):
    """Helicity density h of an ElectromagneticField; sigma w / omega for a plane wave."""
    e, h = field.electric.values, field.magnetic
    overlap = bd.imag(bd.sum(e.conj() * h, axis=-1))

    return -bd.sqrt(material.epsilon * MU0) * overlap / (2 * _omega(field))


def canonical_momentum(jet, material, *, part="dual"):
    """Canonical (orbital) momentum density p_o from an ElectricFieldJet."""
    electric, magnetic = _part(part)
    field = jet.field
    e, h = field.electric.values, field.magnetic
    flow = (
        electric
        * material.epsilon
        * bd.imag(bd.einsum("...j,...ij->...i", e.conj(), jet.gradient))
    )
    flow = flow + magnetic * MU0 * bd.imag(
        bd.einsum("...j,...ij->...i", h.conj(), _magnetic_gradient(jet))
    )

    return flow / (2 * _omega(field))


def spin_momentum(jet, material, *, part="dual"):
    """Spin (Belinfante) momentum density curl S / 2 from an ElectricFieldJet."""
    electric, magnetic = _part(part)
    field = jet.field
    e, h = field.electric.values, field.magnetic
    # d_i S = Im(eps E* x d_i E + mu H* x d_i H) / (2 omega) for the dual parts.
    derivative = (
        electric
        * material.epsilon
        * bd.imag(bd.cross(e.conj()[..., None, :], jet.gradient))
    )
    derivative = derivative + magnetic * MU0 * bd.imag(
        bd.cross(h.conj()[..., None, :], _magnetic_gradient(jet))
    )
    d = derivative / _omega(field)
    curl = bd.stack(
        (
            d[..., 1, 2] - d[..., 2, 1],
            d[..., 2, 0] - d[..., 0, 2],
            d[..., 0, 1] - d[..., 1, 0],
        ),
        axis=-1,
    )

    return curl / 2


def orbital_angular_momentum(jet, material, origin=(0.0, 0.0, 0.0), *, part="dual"):
    """Orbital angular momentum density (r - origin) x p_o about ``origin``."""
    arm = jet.electric.grid.points - bd.asarray(origin, float)

    return bd.cross(arm, canonical_momentum(jet, material, part=part))


def _traction(field, material):
    """T . n on the field's oriented surface quadrature, in N/m²."""
    if not hasattr(field.grid, "weights") or not hasattr(field.grid, "normals"):
        raise ValueError(
            "stress integrals require a closed oriented surface quadrature"
        )

    e, h, n = field.electric.values, field.magnetic, field.grid.normals
    eps = material.epsilon
    w = (eps * bd.sum(bd.abs(e) ** 2, -1) + MU0 * bd.sum(bd.abs(h) ** 2, -1)) / 4
    projected = eps * e * bd.sum(e.conj() * n, -1)[..., None]
    projected = projected + MU0 * h * bd.sum(h.conj() * n, -1)[..., None]

    return 0.5 * bd.real(projected) - w[..., None] * n


def _integrate(density, weights):
    axes = tuple(range(weights.ndim))

    return to_numpy(bd.sum(density * weights[..., None], axis=axes))


def force(field, material):
    """Time-averaged force in newtons on the body inside a closed surface quadrature."""
    return _integrate(_traction(field, material), field.grid.weights)


def torque(field, material, origin=(0.0, 0.0, 0.0)):
    """Time-averaged torque in N m about ``origin`` on the body inside the surface."""
    arm = field.grid.points - bd.asarray(origin, float)

    return _integrate(bd.cross(arm, _traction(field, material)), field.grid.weights)
