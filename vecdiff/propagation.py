import warnings

import numpy as np
from .coordinate_transformation import polar_grid_to_cartesian_grid
from .fresnel import FresnelOvoid
from .fourier import FT2
from .hankel import HankelTransform
from .fields import Field
from .grid import Grid
from .pupil_mapping import (
    pupil_transform,
    radius_from_pupil_coordinate,
    resolve_mapping,
)
from .transfer import interface_operator

exp = np.exp
π = np.pi

# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _sanitize_fresnel(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Replace non-finite Fresnel samples by interpolation from finite neighbours."""
    finite = np.isfinite(t)
    if np.any(finite):
        order = np.argsort(r[finite])
        return np.where(finite, t, np.interp(r, r[finite][order], t[finite][order]))
    return np.zeros_like(r, dtype=float)

def HT(order: int, r: np.ndarray, f_r: np.ndarray, q: np.ndarray) -> np.ndarray:
    return HankelTransform.transform_array(order, r, f_r, q)

def _build_output_grid(field: Field, q: np.ndarray, varphi: np.ndarray) -> Grid:
    """Build the polar output grid in q-space."""
    if field.grid.type == "polar":
        return Grid.from_polar(q, varphi)
    raise ValueError(f"Unsupported grid type: {type(field.grid)}")


def fresnel_coefficients_on_grid(R: np.ndarray, diopter, support=None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(tp, ts)`` Fresnel coefficients sampled on a pupil radius grid.

    ``R`` is the cylindrical radius of the point on the refracting surface --
    the physical aperture coordinate.  If ``support`` is provided, coefficients
    are evaluated only where the support mask is true and are zero elsewhere.

    These are the bare interface coefficients.  The weights the propagators
    actually apply also carry the geometric factor of Eq. (47) and the
    meridional projection; see :func:`transfer_weights_on_grid`.
    """
    R = np.asarray(R, dtype=float)
    tp = np.zeros_like(R, dtype=float)
    ts = np.zeros_like(R, dtype=float)

    if support is None:
        active = np.ones(R.shape, dtype=bool)
    else:
        active = np.asarray(support, dtype=bool)
        if active.shape != R.shape:
            raise ValueError("support must have the same shape as R.")

    if not np.any(active):
        return tp, ts

    fresnel = FresnelOvoid(ovoid=diopter)
    r_active = R[active]
    with np.errstate(divide="ignore", invalid="ignore"):
        tp_active = fresnel.tp(r_active)
        ts_active = fresnel.ts(r_active)
    tp[active] = _sanitize_fresnel(r_active, tp_active)
    ts[active] = _sanitize_fresnel(r_active, ts_active)
    return tp, ts


def transfer_weights_on_grid(
    R: np.ndarray, diopter, support=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(lam_r, lam_phi, lam_z)`` sampled on a pupil radius grid.

    These are the eigenvalues of Eqs. (62)-(63): the Fresnel coefficients times
    the geometric amplitude factor ``A(Q)``, with the meridional projection
    ``cos(alpha_i) / cos(alpha_0)`` on the radial channel.
    """
    R = np.asarray(R, dtype=float)
    lam_r = np.zeros_like(R, dtype=float)
    lam_phi = np.zeros_like(R, dtype=float)
    lam_z = np.zeros_like(R, dtype=float)

    if support is None:
        active = np.ones(R.shape, dtype=bool)
    else:
        active = np.asarray(support, dtype=bool)
        if active.shape != R.shape:
            raise ValueError("support must have the same shape as R.")

    if not np.any(active):
        return lam_r, lam_phi, lam_z

    r_active = R[active]
    operator = interface_operator(diopter, r_active)
    a_r, a_phi, a_z = operator.eigenvalues()

    lam_r[active] = _sanitize_fresnel(r_active, a_r)
    lam_phi[active] = _sanitize_fresnel(r_active, a_phi)
    lam_z[active] = _sanitize_fresnel(r_active, a_z)
    return lam_r, lam_phi, lam_z


def _warn_if_aperture_clips(Ex, Ey, R, diopter, *, tolerance: float = 1e-4) -> None:
    """Warn when a meaningful share of the field falls past the usable aperture.

    Measured in energy rather than in sample count: after a free-space
    propagation step the field is numerically non-zero across the whole grid,
    so counting non-zero samples outside the aperture reports almost the entire
    grid and says nothing about whether anything was actually lost.
    """
    limit = getattr(diopter, "aperture_limit", None)
    if limit is None:
        return  # weights are being injected; there is no surface to clip against
    energy = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
    total = float(energy.sum())
    if total <= 0.0:
        return
    outside = float(energy[R > limit].sum()) / total
    if outside > tolerance:
        warnings.warn(
            f"{100.0 * outside:.1f}% of the incident energy lies past the surface's usable "
            f"aperture (r > {limit:.6g}) and cannot be transmitted. "
            "Reduce the aperture, or check whether the radius being passed is the "
            "cylindrical radius on the refracting surface.",
            RuntimeWarning,
            stacklevel=3,
        )


def transverse_diopter_operator(
    field: Field, diopter, *, radius: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the local transverse diopter operator to a Cartesian field.

    The algebra is Eq. (67): a scalar part plus a ``cos 2phi`` / ``sin 2phi``
    quadrupole.  What changed with the geometric correction is the pair of
    eigenvalues feeding it, not the shape of the operator.

    ``radius`` overrides where the transfer weights are sampled.  It is needed
    once the field has been warped onto the transform variable ``u``: the
    samples then sit at ``u``, but the weights still belong to the surface
    radius ``r`` the sample came from.  The warp is purely radial, so the
    azimuth the quadrupole uses is unaffected.
    """
    if field.grid.type != "cartesian":
        raise ValueError("transverse_diopter_operator requires a Cartesian grid.")

    Ex = np.asarray(field.x, dtype=complex)
    Ey = np.asarray(field.y, dtype=complex)
    if Ex.shape != Ey.shape or Ex.shape != field.grid.shape:
        raise ValueError("field components must match the Cartesian grid shape.")

    R = field.grid.R if radius is None else np.asarray(radius, dtype=float)
    if R.shape != Ex.shape:
        raise ValueError("radius must have the same shape as the field components.")

    support = (np.abs(Ex) > 0.0) | (np.abs(Ey) > 0.0)
    _warn_if_aperture_clips(Ex, Ey, R, diopter)
    lam_r, lam_phi, _ = transfer_weights_on_grid(
        R, diopter, support=support
    )
    t_plus = 0.5 * (lam_r + lam_phi)
    t_minus = 0.5 * (lam_r - lam_phi)

    c2 = np.cos(2.0 * field.grid.Phi)
    s2 = np.sin(2.0 * field.grid.Phi)

    Ux = (t_plus + t_minus * c2) * Ex + (t_minus * s2) * Ey
    Uy = (t_minus * s2) * Ex + (t_plus - t_minus * c2) * Ey
    return Ux, Uy


def _polar_component_to_cartesian_grid(component, radial_axis, angular_axis, X, Y):
    component = np.asarray(component, dtype=complex)
    rho = np.sqrt(X**2 + Y**2)

    if component.ndim == 1:
        return np.interp(rho, radial_axis, component, left=component[0], right=0.0)

    if component.ndim == 2 and component.shape[0] == 1:
        row = component[0]
        return np.interp(rho, radial_axis, row, left=row[0], right=0.0)

    return polar_grid_to_cartesian_grid(
        component,
        radial_axis,
        angular_axis,
        X,
        Y,
        fill_value=0.0,
    )


def _field_with_cartesian_grid_for_fft(field: Field) -> Field:
    if field.grid.type == "cartesian":
        return field
    if field.grid.type != "polar":
        raise ValueError("FFT diopter propagation requires a Cartesian or polar input grid.")

    print("A Cartesian Grid is preferable for FFT diopter propagation; converting the polar Grid to a Cartesian Grid.")

    r = np.asarray(field.grid.r, dtype=float)
    varphi = np.asarray(field.grid.varphi, dtype=float)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("polar grid must provide at least two radial samples.")
    if varphi.ndim != 1 or varphi.size < 2:
        raise ValueError("polar grid must provide at least two angular samples.")

    half_size = float(np.max(r))
    n_cart = max(int(r.size), 2)
    dx = 2.0 * half_size / n_cart
    cart_grid = Grid.from_spacing((n_cart, n_cart), dx=dx, dy=dx)
    X, Y = cart_grid.X, cart_grid.Y

    Ex = _polar_component_to_cartesian_grid(field.x, r, varphi, X, Y)
    Ey = _polar_component_to_cartesian_grid(field.y, r, varphi, X, Y)
    Ez = None
    if field.z is not None:
        Ez = _polar_component_to_cartesian_grid(field.z, r, varphi, X, Y)
    return Field.from_cartesian(Ex, Ey, cart_grid, symmetric=False, Ez=Ez)


def _warp_to_pupil_grid(field: Field, diopter, mapping: str) -> tuple[Field, np.ndarray]:
    """Resample a Cartesian pupil field onto the transform variable ``u``.

    The Debye reduction integrates over ``u = |zi| sin(alpha_i)``, which is a
    non-linear function of the surface radius ``r``.  The Hankel path can
    integrate over a non-uniform abscissa directly; the FFT path cannot, so the
    field is warped radially onto a uniform ``u`` grid first.  The warp is
    purely radial, so it is a one-dimensional lookup applied to the sample
    positions.

    Returns the warped field together with the surface radius each ``u`` sample
    came from, which is where the transfer weights must be evaluated.
    """
    from scipy.interpolate import RegularGridInterpolator

    grid = field.grid
    x_axis = np.asarray(grid.X[0, :], dtype=float)
    y_axis = np.asarray(grid.Y[:, 0], dtype=float)

    support = (np.abs(field.x) > 0.0) | (np.abs(field.y) > 0.0)
    r_support = float(np.max(grid.R[support])) if np.any(support) else float(np.max(grid.R))
    r_max = min(r_support, diopter.aperture_limit)

    geom_edge = diopter.ray_geometry(np.array([r_max]))
    u_max = float(pupil_transform(geom_edge, mapping)[0][0])

    ny, nx = grid.shape
    # Keep the sample count, span the mapped aperture.
    ux = np.linspace(-u_max, u_max, nx)
    uy = np.linspace(-u_max, u_max, ny)
    u_grid = Grid.from_axes(ux, uy, domain=grid.domain, reference=grid.reference)

    r_of_u, inside = radius_from_pupil_coordinate(diopter, u_grid.R, mapping)
    with np.errstate(divide="ignore", invalid="ignore"):
        stretch = np.divide(
            r_of_u,
            u_grid.R,
            out=np.ones_like(u_grid.R),
            where=u_grid.R > 0.0,
        )
    X_src = stretch * u_grid.X
    Y_src = stretch * u_grid.Y
    points = np.column_stack((Y_src.ravel(), X_src.ravel()))

    def resample(component):
        if component is None:
            return None
        interp = RegularGridInterpolator(
            (y_axis, x_axis), np.asarray(component, dtype=complex),
            bounds_error=False, fill_value=0.0,
        )
        return np.where(inside, interp(points).reshape(u_grid.shape), 0.0)

    warped = Field.from_cartesian(
        resample(field.x),
        resample(field.y),
        u_grid,
        symmetric=field.symmetry == "cartesian",
        Ez=resample(field.z),
    )
    return warped, np.where(inside, r_of_u, np.inf)


def _validate_pad_factor(pad_factor: int) -> int:
    if isinstance(pad_factor, bool) or not hasattr(pad_factor, "__index__"):
        raise TypeError("pad_factor must be an integer greater than or equal to 1.")
    pad_factor = pad_factor.__index__()
    if pad_factor < 1:
        raise ValueError("pad_factor must be greater than or equal to 1.")
    return pad_factor


def _center_pad_array(values: np.ndarray, pad_factor: int) -> np.ndarray:
    values = np.asarray(values, dtype=complex)
    if pad_factor == 1:
        return values

    ny, nx = values.shape
    target_y = ny * pad_factor
    target_x = nx * pad_factor
    pad_y = target_y - ny
    pad_x = target_x - nx
    padding = (
        (pad_y // 2, pad_y - pad_y // 2),
        (pad_x // 2, pad_x - pad_x // 2),
    )
    return np.pad(values, padding, mode="constant")


def _center_padded_grid(grid: Grid, pad_factor: int) -> Grid:
    if pad_factor == 1:
        return grid

    dx, dy = grid.spacing()
    padded_shape = (
        grid.shape[0] * pad_factor,
        grid.shape[1] * pad_factor,
    )
    return Grid.from_spacing(padded_shape, dx=float(dx), dy=float(dy), domain=grid.domain)


def propagate_to_focal_plane_through_diopter_fft(
    field: Field,
    diopter,
    *,
    include_prefactor: bool = False,
    wavelength: float | None = None,
    output: str = "k",
    pad_factor: int = 1,
    kgrid: Grid | None = None,
    ft_method: str = "auto",
    transmission: str = "vectorial",
    mapping: str | None = None,
) -> Field:
    """Propagate an arbitrary Cartesian field through a diopter using FFT2.

    ``pad_factor`` zero-pads the post-diopter field before the FFT, refining
    the sampled k/focal grid without changing the input sampling interval.

    ``mapping`` selects the pupil variable the transform integrates over and
    defaults to the Debye sine mapping.  See :mod:`vecdiff.pupil_mapping`.
    """
    field = _field_with_cartesian_grid_for_fft(field)
    pad_factor = _validate_pad_factor(pad_factor)
    if output not in {"k", "focal"}:
        raise ValueError("output must be either 'k' or 'focal'.")
    if output == "focal" and wavelength is None:
        raise ValueError("wavelength is required when output='focal'.")
    if include_prefactor and wavelength is None:
        raise ValueError("wavelength is required when include_prefactor=True.")
    if transmission not in {"vectorial", "identity"}:
        raise ValueError("transmission must be either 'vectorial' or 'identity'.")
    mapping = resolve_mapping(mapping)

    # Validates that the grid is uniform and obtains the physical sampling.
    field.grid.spacing()

    field = incident_field_on_sphere(field, diopter)
    field, r_of_sample = _warp_to_pupil_grid(field, diopter, mapping)
    reachable = np.isfinite(r_of_sample)
    r_weights = np.where(reachable, r_of_sample, 0.0)

    if transmission == "vectorial":
        Ux, Uy = transverse_diopter_operator(
            field, diopter, radius=r_weights
        )
    else:
        Ux = np.asarray(field.x, dtype=complex)
        Uy = np.asarray(field.y, dtype=complex)

    _, weight = pupil_transform(diopter.ray_geometry(r_weights), mapping)
    weight = np.where(reachable, weight, 0.0)
    Ux = weight * Ux
    Uy = weight * Uy

    fft_grid = _center_padded_grid(field.grid, pad_factor)
    Ux_fft = _center_pad_array(Ux, pad_factor)
    Uy_fft = _center_pad_array(Uy, pad_factor)
    requested_kgrid = kgrid
    Ex_out, kgrid = FT2(Ux_fft, fft_grid, kgrid=requested_kgrid, physical=True, method=ft_method)
    Ey_out, _ = FT2(Uy_fft, fft_grid, kgrid=requested_kgrid, physical=True, method=ft_method)

    if include_prefactor:
        from .pupil_mapping import debye_prefactor

        prefactor = debye_prefactor(diopter, wavelength) / (2.0 * π)
        Ex_out = prefactor * Ex_out
        Ey_out = prefactor * Ey_out

    if output == "k":
        grid_out = kgrid
    else:
        scale = wavelength * abs(diopter.zi) / (2.0 * π * diopter.ni)
        grid_out = Grid.from_cartesian(
            scale * kgrid.X,
            scale * kgrid.Y,
            domain="focal",
            dual=field.grid,
        )

    return Field.from_cartesian(Ex_out, Ey_out, grid_out, symmetric=False)

# ------------------------------------------------------------------ #
#  Input geometry                                                      #
# ------------------------------------------------------------------ #

def incident_field_on_sphere(field: Field, diopter) -> Field:
    """Return the field re-expressed on the incident reference sphere ``G``.

    A field is the pair ``(r, E(r))``, and the propagators need to know which
    surface the samples live on; ``Grid.reference`` carries that.

    * ``None`` (the default) or the incident sphere ``G``: the samples are
      already labelled by the surface radius ``r``, which is the same thing --
      a point of ``G`` is labelled by its ray, and rays are labelled by ``r``.
      Returned unchanged.
    * The tangent plane ``P``: the rays from ``A`` cross ``P`` at
      ``r_P = |z0| tan(alpha_0)`` (Eq. 85), and equating the flux through the
      plane with the flux through ``G`` gives ``E_G = E_P / |cos(alpha_0)|``
      (Eq. 86).  Both the radial remap and the factor are applied.

    Unlike the focal-plane step, this one genuinely is a plane-to-sphere flux
    relation, so the perspective mapping is the right one here.
    """
    from .geometry import ReferenceSphere, TangentPlane

    reference = getattr(field.grid, "reference", None)
    if reference is None:
        return field

    if isinstance(reference, ReferenceSphere):
        expected = ReferenceSphere(center_z=float(diopter.z0), radius=abs(float(diopter.z0)))
        if reference == expected:
            return field
        raise ValueError(
            f"field lives on {reference}, which is not the incident sphere {expected} "
            "of this diopter."
        )

    if isinstance(reference, TangentPlane):
        return _tangent_plane_to_incident_sphere(field, diopter)

    raise ValueError(f"unsupported grid reference: {reference!r}")


def _tangent_plane_to_incident_sphere(field: Field, diopter) -> Field:
    """Remap ``r_P -> r`` and apply the ``1/|cos(alpha_0)|`` flux factor."""
    grid = field.grid
    if grid.type == "polar":
        r_plane = np.asarray(grid.r, dtype=float)
    else:
        r_plane = np.asarray(grid.R, dtype=float)

    r_surface, inside = _object_plane_radius_to_surface(diopter, r_plane)
    geom = diopter.ray_geometry(np.where(inside, r_surface, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.divide(
            1.0,
            np.abs(geom.cos_a0),
            out=np.zeros_like(geom.r),
            where=np.abs(geom.cos_a0) > 0.0,
        )
    factor = np.where(inside & geom.valid, factor, 0.0)

    if grid.type == "polar":
        new_grid = Grid.from_polar(r_surface, grid.varphi)
        broadcast = factor if factor.ndim == 2 else factor[None, :]
    else:
        raise NotImplementedError(
            "plane-referenced input is currently supported on polar grids only."
        )

    return Field.from_cartesian(
        field.x * broadcast,
        field.y * broadcast,
        new_grid,
        symmetric=field.symmetry == "cartesian",
        Ez=None if field.z is None else field.z * broadcast,
    )


def _object_plane_radius_to_surface(diopter, r_plane, *, n_table: int = 20_001):
    """Invert ``r_P = |z0| tan(alpha_0)`` for the surface radius ``r``."""
    r_plane = np.asarray(r_plane, dtype=float)
    r_table = np.linspace(0.0, diopter.aperture_limit, int(n_table))
    geom = diopter.ray_geometry(r_table)
    plane_table = geom.r_tan_obj

    keep = np.concatenate(([True], np.diff(plane_table) > 0.0))
    plane_table, r_table = plane_table[keep], r_table[keep]

    inside = (r_plane >= 0.0) & (r_plane <= plane_table[-1])
    r = np.interp(np.clip(r_plane, 0.0, plane_table[-1]), plane_table, r_table)
    return r, inside


# ------------------------------------------------------------------ #
#  Propagation                                                         #
# ------------------------------------------------------------------ #

def propagate_to_focal_plane_through_diopter(
    field: Field,
    diopter,
    q: np.ndarray,
    *,
    mapping: str | None = None,
) -> Field:
    """Propagate an axially symmetric pupil field to the focal plane by Hankel transform.

    The pupil grid carries the surface radius ``r``; the transform integrates
    over ``u(r)`` set by ``mapping``.  Unlike the FFT path this needs no
    resampling: the quadrature accepts a non-uniform abscissa, so ``u(r)`` is
    passed straight in as the integration variable.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 1:
        raise ValueError("q must be a 1D array of output radial samples.")
    if q.size < 2:
        raise ValueError("q must contain at least two samples.")
    if not np.all(np.diff(q) > 0):
        raise ValueError("q must be strictly increasing.")
    if field.grid.type != "polar":
        raise NotImplementedError("Focal-plane Hankel propagation requires a polar input grid.")
    mapping = resolve_mapping(mapping)

    field = incident_field_on_sphere(field, diopter)

    r = np.asarray(getattr(field.grid, "r", field.grid.R), dtype=float).ravel()
    varphi = np.asarray(getattr(field.grid, "varphi", np.array([0.0])), dtype=float).ravel()
    if r.ndim != 1 or r.size < 2:
        raise ValueError("field.grid must provide a 1D radial input grid with at least two samples.")
    if field.grid.type == "polar" and varphi.size < 1:
        raise ValueError("field.grid must provide at least one angular sample.")
    varphi_col = varphi.reshape(-1, 1)

    operator = interface_operator(diopter, r)
    lam_r, lam_phi, _ = operator.eigenvalues()
    lam_r = _sanitize_fresnel(r, lam_r)
    lam_phi = _sanitize_fresnel(r, lam_phi)

    u, weight = pupil_transform(diopter.ray_geometry(r), mapping)
    tp = lam_r * weight
    ts = lam_phi * weight

    grid_out = _build_output_grid(field, q, varphi)

    if field.symmetry == 'circular':
        h0L = HT(0, u, (tp + ts) * field.L, q)
        h2L = HT(2, u, (tp - ts) * field.R, q)
        h0R = HT(0, u, (tp + ts) * field.R, q)
        h2R = HT(2, u, (tp - ts) * field.L, q)
        E_L = (π*h0L)[None, :] - π*exp( 2.0j * varphi_col) * h2L[None, :]
        E_R = (π*h0R)[None, :] - π*exp(-2.0j * varphi_col) * h2R[None, :]
        return Field.from_circular(E_L, E_R, grid_out, symmetric=True)

    if field.symmetry == 'polar':
        h1r   = HT(1, u, tp * field.r,   q)
        h1phi = HT(1, u, ts * field.phi, q)
        E_r   = np.broadcast_to((-2.0j*π*h1r  )[None, :], (len(varphi), len(q))).copy()
        E_phi = np.broadcast_to((-2.0j*π*h1phi)[None, :], (len(varphi), len(q))).copy()
        return Field.from_polar(E_r, E_phi, grid_out, symmetric=True)

    if field.symmetry == 'cartesian':
        h0x = HT(0, u, (tp + ts) * field.x, q)
        h2x = HT(2, u, (tp - ts) * field.x, q)
        h0y = HT(0, u, (tp + ts) * field.y, q)
        h2y = HT(2, u, (tp - ts) * field.y, q)
        c2  = np.cos(2 * varphi_col)
        s2  = np.sin(2 * varphi_col)
        E_x = π * (h0x[None, :] - c2 * h2x[None, :] - s2 * h2y[None, :])
        E_y = π * (h0y[None, :] + c2 * h2y[None, :] - s2 * h2x[None, :])
        return Field.from_cartesian(E_x, E_y, grid_out, symmetric=True)

    if field.symmetry is None:
        raise NotImplementedError("Propagation requires a field with explicit symmetry.")
    raise ValueError(f"Unsupported field symmetry: {field.symmetry!r}")
