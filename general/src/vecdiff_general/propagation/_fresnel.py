"""Vector Fresnel coefficients and their fused application.

The fused symbol retains complex transmitted directions under TIR. Its
absolute-incidence-cosine continuation at back-facing chart points belongs
to the forward approximation; it does not solve closed-body shadowing."""

from vecdiff_general._arrays import positive
from vecdiff_general._backend import bd
from vecdiff_general.propagation._directions import reflected_direction, unit_direction


def cosines_from_directions(incident_direction, transmitted_direction, normal):
    """Incidence/transmission cosines for a normal toward transmitted matter."""
    incident_direction = unit_direction(incident_direction)
    normal = unit_direction(normal)
    transmitted_direction = bd.asarray(transmitted_direction, dtype=complex)

    if (
        transmitted_direction.shape[-1:] != (3,)
        or not bd.isfinite(transmitted_direction).all()
    ):
        raise ValueError("transmitted directions must be finite Cartesian vectors")

    bilinear_norm = bd.sum(transmitted_direction * transmitted_direction, axis=-1)

    if not bd.allclose(bilinear_norm, 1, rtol=1e-11, atol=1e-12):
        raise ValueError("transmitted directions must satisfy u dot u = 1")

    cos_incidence = bd.sum(incident_direction * normal, axis=-1)
    cos_transmission = bd.sum(transmitted_direction * normal, axis=-1)

    return cos_incidence, cos_transmission


def _coefficient_arguments(
    incident_index, transmitted_index, cos_incidence, cos_transmission
):
    incident_index = positive(incident_index, "incident index")
    transmitted_index = positive(transmitted_index, "transmitted index")

    cos_incidence, cos_transmission = bd.broadcast_arrays(
        bd.asarray(cos_incidence, dtype=complex),
        bd.asarray(cos_transmission, dtype=complex),
    )

    if not bd.isfinite(cos_incidence).all() or not bd.isfinite(cos_transmission).all():
        raise ValueError("Fresnel cosines must be finite")

    if bd.any(bd.abs(cos_incidence.imag) > 1e-12) or bd.any(
        cos_incidence.real <= 1e-12
    ):
        raise ValueError("incidence must be propagating and away from grazing")

    if bd.any(cos_transmission.real < -1e-12) or bd.any(cos_transmission.imag < -1e-12):
        raise ValueError("transmission must use the outgoing or decaying branch")

    if bd.any(bd.abs(cos_transmission) <= 1e-12):
        raise ValueError("exact critical incidence requires a limiting representation")

    return incident_index, transmitted_index, cos_incidence, cos_transmission


def transmission_coefficients(
    incident_index, transmitted_index, cos_incidence, cos_transmission
):
    """Return only the electric transmission amplitudes (t_s, t_p)."""
    incident_index, transmitted_index, cos_incidence, cos_transmission = (
        _coefficient_arguments(
            incident_index, transmitted_index, cos_incidence, cos_transmission
        )
    )

    denominator_s = (
        incident_index * cos_incidence + transmitted_index * cos_transmission
    )
    denominator_p = (
        transmitted_index * cos_incidence + incident_index * cos_transmission
    )

    transmission_s = 2 * incident_index * cos_incidence / denominator_s
    transmission_p = 2 * incident_index * cos_incidence / denominator_p

    return transmission_s, transmission_p


def reflection_coefficients(
    incident_index, transmitted_index, cos_incidence, cos_transmission
):
    """Return only electric reflection amplitudes (r_s, r_p) in branch s/p bases."""
    incident_index, transmitted_index, cos_incidence, cos_transmission = (
        _coefficient_arguments(
            incident_index, transmitted_index, cos_incidence, cos_transmission
        )
    )

    denominator_s = (
        incident_index * cos_incidence + transmitted_index * cos_transmission
    )
    denominator_p = (
        transmitted_index * cos_incidence + incident_index * cos_transmission
    )

    reflection_s = (
        incident_index * cos_incidence - transmitted_index * cos_transmission
    ) / denominator_s
    reflection_p = (
        transmitted_index * cos_incidence - incident_index * cos_transmission
    ) / denominator_p

    return reflection_s, reflection_p


def polarization_basis(incident_direction, output_direction, normal):
    """Return (s, p_incident, p_output), with a regular normal-incidence basis.

    The output may be reflected, transmitted, or complex under TIR. Bilinear
    polarization products are used; a complex p vector is not renormalized
    with a Hermitian norm. At normal incidence the arbitrary s choice cancels
    from the Cartesian operator.
    """
    incident_direction = unit_direction(incident_direction)
    normal = unit_direction(normal)
    output_direction = bd.asarray(output_direction, dtype=complex)

    incident_direction, output_direction, normal = bd.broadcast_arrays(
        incident_direction, output_direction, normal
    )

    s_vector = bd.cross(incident_direction, normal)
    s_length = bd.linalg.norm(s_vector, axis=-1)

    auxiliary_axis = bd.eye(3)[bd.argmin(bd.abs(incident_direction), axis=-1)]
    fallback = bd.cross(incident_direction, auxiliary_axis)
    fallback /= bd.linalg.norm(fallback, axis=-1)[..., None]

    resolved = (s_length > 1e-12)[..., None]
    s_vector = bd.where(
        resolved,
        s_vector / bd.where(resolved, s_length[..., None], 1.0),
        fallback,
    )

    incident_p = bd.cross(s_vector, incident_direction)
    output_p = bd.cross(s_vector, output_direction)

    return s_vector, incident_p, output_p


def fresnel_transmission_matrix(
    incident_direction, transmitted_direction, normal, incident_index, transmitted_index
):
    """Cartesian transmission operator T = t_p p_t ⊗ p_i + t_s s ⊗ s."""
    cos_incidence, cos_transmission = cosines_from_directions(
        incident_direction, transmitted_direction, normal
    )

    transmission_s, transmission_p = transmission_coefficients(
        incident_index, transmitted_index, cos_incidence, cos_transmission
    )

    s_vector, incident_p, transmitted_p = polarization_basis(
        incident_direction, transmitted_direction, normal
    )

    p_projection = transmitted_p[..., :, None] * incident_p[..., None, :]
    s_projection = s_vector[..., :, None] * s_vector[..., None, :]

    return (
        transmission_p[..., None, None] * p_projection
        + transmission_s[..., None, None] * s_projection
    )


def fresnel_reflection_matrix(
    incident_direction, transmitted_direction, normal, incident_index, transmitted_index
):
    """Cartesian reflection operator R = r_p p_r ⊗ p_i + r_s s ⊗ s.

    The transmitted direction supplies the transmission cosine needed by the
    reflection law, including TIR. No transmitted field or coefficient is made.
    """
    cos_incidence, cos_transmission = cosines_from_directions(
        incident_direction, transmitted_direction, normal
    )

    reflection_s, reflection_p = reflection_coefficients(
        incident_index, transmitted_index, cos_incidence, cos_transmission
    )

    reflection_direction = reflected_direction(incident_direction, normal)

    s_vector, incident_p, reflected_p = polarization_basis(
        incident_direction, reflection_direction, normal
    )

    p_projection = reflected_p[..., :, None] * incident_p[..., None, :]
    s_projection = s_vector[..., :, None] * s_vector[..., None, :]

    return (
        reflection_p[..., None, None] * p_projection
        + reflection_s[..., None, None] * s_projection
    )


def apply_fresnel(electric, matrix):
    """Apply a Cartesian operator to electric arrays; supports (..., 3) samples."""
    electric = bd.asarray(electric, dtype=complex)
    matrix = bd.asarray(matrix, dtype=complex)

    if electric.shape[-1:] != (3,) or matrix.shape[-2:] != (3, 3):
        raise ValueError("expected Cartesian vectors and 3 by 3 operators")

    if not bd.isfinite(electric).all() or not bd.isfinite(matrix).all():
        raise ValueError("electric samples and Fresnel operators must be finite")

    return bd.einsum("...ij,...j->...i", matrix, electric)


def _apply_to_field(field, matrix, incident_direction):
    from vecdiff_general.fields.electromagnetic_field import ElectricField

    if not isinstance(field, ElectricField):
        raise TypeError("an interface operator requires a complete ElectricField")

    incident_direction = unit_direction(incident_direction)
    longitudinal = bd.abs(bd.sum(incident_direction * field.values, axis=-1))
    amplitude = bd.linalg.norm(field.values, axis=-1)

    if bd.any(longitudinal > 1e-10 * amplitude):
        raise ValueError(
            "incident field must be transverse to its prescribed directions"
        )

    values = apply_fresnel(field.values, matrix)

    return ElectricField(field.grid, field.wavelength0, values)


def transmit(
    field,
    incident_direction,
    transmitted_direction,
    normal,
    incident_index,
    transmitted_index,
):
    """Transmit a locally directed electric field on its existing surface grid.

    A multi-mode incident field requires applying this operator to each mode
    before coherent summation. No unique ray is inferred from a total field.
    """
    matrix = fresnel_transmission_matrix(
        incident_direction,
        transmitted_direction,
        normal,
        incident_index,
        transmitted_index,
    )

    return _apply_to_field(field, matrix, incident_direction)


def reflect(
    field,
    incident_direction,
    transmitted_direction,
    normal,
    incident_index,
    transmitted_index,
):
    """Reflect a locally directed electric field on its existing surface grid."""
    matrix = fresnel_reflection_matrix(
        incident_direction,
        transmitted_direction,
        normal,
        incident_index,
        transmitted_index,
    )

    return _apply_to_field(field, matrix, incident_direction)


def _fresnel_basis(d, n, n1, n2):
    ci = bd.sum(d * n, axis=-1)
    s = bd.cross(d, n)
    size = bd.linalg.norm(s, axis=-1, keepdims=True)
    fallback = bd.cross(
        n,
        bd.where(
            abs(n[..., :1]) < 0.9, bd.asarray([1.0, 0, 0]), bd.asarray([0, 1.0, 0])
        ),
    )
    s = bd.where(size > 1e-12, s / bd.maximum(size, 1e-300), fallback)
    s /= bd.linalg.norm(s, axis=-1, keepdims=True)
    ci_abs = abs(ci)
    ct = bd.sqrt(1 - (n1 / n2) ** 2 * (1 - ci_abs**2) + 0j)
    dt = (n1 / n2) * (d - ci[..., None] * n) + (bd.where(ci < 0, -1.0, 1.0) * ct)[
        ..., None
    ] * n
    pi, pt = bd.cross(s, d), bd.cross(s, dt)
    ts = 2 * n1 * ci_abs / (n1 * ci_abs + n2 * ct)
    tp = 2 * n1 * ci_abs / (n2 * ci_abs + n1 * ct)
    return s, pi, pt, dt, ts, tp


def apply_transmission(directions, normals, n1, n2, wave, *, coefficients="fresnel"):
    """Apply transmission with Fresnel or unit s/p electric amplitudes.

    Unit amplitudes keep Snell refraction and the incident/transmitted
    polarization bases. They isolate Fresnel weighting in comparisons.
    """
    if coefficients not in ("fresnel", "unity"):
        raise ValueError("coefficients must be 'fresnel' or 'unity'")
    d, n = bd.broadcast_arrays(bd.asarray(directions), bd.asarray(normals))
    wave = bd.asarray(wave)
    ez = -(d[..., 0] * wave[..., 0] + d[..., 1] * wave[..., 1]) / d[..., 2]
    if n1 == n2:
        electric = bd.concatenate((wave, ez[..., None]), axis=-1)
        return bd.concatenate((electric, n1 * bd.cross(d, electric)), axis=-1)
    s, pi, pt, dt, ts, tp = _fresnel_basis(d, n, n1, n2)
    if coefficients == "unity":
        ts, tp = bd.ones_like(ts), bd.ones_like(tp)
    a_s = s[..., 0] * wave[..., 0] + s[..., 1] * wave[..., 1] + s[..., 2] * ez
    a_p = pi[..., 0] * wave[..., 0] + pi[..., 1] * wave[..., 1] + pi[..., 2] * ez
    electric = (ts * a_s)[..., None] * s + (tp * a_p)[..., None] * pt
    return bd.concatenate((electric, n2 * bd.cross(dt, electric)), axis=-1)
