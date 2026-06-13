import numpy as np
from .fresnel import FresnelOvoid
from .fourier import FT2
from .hankel import HankelTransform
from .fields import Field
from .grid import Grid

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
    """Return ``(tp, ts)`` Fresnel coefficients sampled on a Cartesian radial grid.

    If ``support`` is provided, coefficients are evaluated only where the support
    mask is true and are set to zero elsewhere.
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


def transverse_diopter_operator(field: Field, diopter) -> tuple[np.ndarray, np.ndarray]:
    """Apply the local transverse diopter operator to a Cartesian field."""
    if field.grid.type != "cartesian":
        raise ValueError("transverse_diopter_operator requires a Cartesian grid.")

    Ex = np.asarray(field.x, dtype=complex)
    Ey = np.asarray(field.y, dtype=complex)
    if Ex.shape != Ey.shape or Ex.shape != field.grid.shape:
        raise ValueError("field components must match the Cartesian grid shape.")

    support = (np.abs(Ex) > 0.0) | (np.abs(Ey) > 0.0)
    tp, ts = fresnel_coefficients_on_grid(field.grid.R, diopter, support=support)
    t_plus = 0.5 * (tp + ts)
    t_minus = 0.5 * (tp - ts)

    c2 = np.cos(2.0 * field.grid.Phi)
    s2 = np.sin(2.0 * field.grid.Phi)

    Ux = (t_plus + t_minus * c2) * Ex + (t_minus * s2) * Ey
    Uy = (t_minus * s2) * Ex + (t_plus - t_minus * c2) * Ey
    return Ux, Uy


def propagate_to_focal_plane_through_diopter_fft(
    field: Field,
    diopter,
    *,
    include_prefactor: bool = False,
    wavelength: float | None = None,
    output: str = "k",
) -> Field:
    """Propagate an arbitrary Cartesian field through a diopter using FFT2."""
    if field.grid.type != "cartesian":
        raise ValueError("FFT diopter propagation requires a Cartesian input grid.")
    if output not in {"k", "focal"}:
        raise ValueError("output must be either 'k' or 'focal'.")
    if output == "focal" and wavelength is None:
        raise ValueError("wavelength is required when output='focal'.")
    if include_prefactor and wavelength is None:
        raise ValueError("wavelength is required when include_prefactor=True.")

    # Validates that the grid is uniform and obtains the physical sampling.
    field.grid.spacing()

    Ux, Uy = transverse_diopter_operator(field, diopter)
    Ex_out, kgrid = FT2(Ux, field.grid, physical=True)
    Ey_out, _ = FT2(Uy, field.grid, physical=True)

    if include_prefactor:
        k0 = 2.0 * π / wavelength
        prefactor = diopter.ni * exp(1.0j * diopter.ni * k0 * diopter.zi)
        prefactor /= 1.0j * wavelength * diopter.zi * diopter.z0
        Ex_out = prefactor * Ex_out
        Ey_out = prefactor * Ey_out

    if output == "k":
        grid_out = kgrid
    else:
        scale = wavelength * diopter.zi / (2.0 * π * diopter.ni)
        grid_out = Grid.from_cartesian(
            scale * kgrid.X,
            scale * kgrid.Y,
            domain="focal",
            dual=field.grid,
        )

    return Field.from_cartesian(Ex_out, Ey_out, grid_out, symmetric=False)

# ------------------------------------------------------------------ #
#  Propagation                                                         #
# ------------------------------------------------------------------ #

def propagate_to_focal_plane_through_diopter(field: Field, diopter, q: np.ndarray) -> Field:
    q = np.asarray(q, dtype=float)
    if q.ndim != 1:
        raise ValueError("q must be a 1D array of output radial samples.")
    if q.size < 2:
        raise ValueError("q must contain at least two samples.")
    if not np.all(np.diff(q) > 0):
        raise ValueError("q must be strictly increasing.")
    if field.grid.type != "polar":
        raise NotImplementedError("Focal-plane Hankel propagation requires a polar input grid.")

    r = np.asarray(getattr(field.grid, "r", field.grid.R), dtype=float).ravel()
    varphi = np.asarray(getattr(field.grid, "varphi", np.array([0.0])), dtype=float).ravel()
    if r.ndim != 1 or r.size < 2:
        raise ValueError("field.grid must provide a 1D radial input grid with at least two samples.")
    if field.grid.type == "polar" and varphi.size < 1:
        raise ValueError("field.grid must provide at least one angular sample.")
    varphi_col = varphi.reshape(-1, 1)

    fresnel = FresnelOvoid(ovoid=diopter)
    with np.errstate(divide="ignore", invalid="ignore"):
        tp = _sanitize_fresnel(r, fresnel.tp(r))
        ts = _sanitize_fresnel(r, fresnel.ts(r))

    grid_out = _build_output_grid(field, q, varphi)

    if field.symmetry == 'circular':
        h0L = HT(0, r, (tp + ts) * field.L, q)
        h2L = HT(2, r, (tp - ts) * field.R, q)
        h0R = HT(0, r, (tp + ts) * field.R, q)
        h2R = HT(2, r, (tp - ts) * field.L, q)
        E_L = (2*π*h0L)[None, :] - 2*π*exp( 2.0j * varphi_col) * h2L[None, :]
        E_R = (2*π*h0R)[None, :] - 2*π*exp(-2.0j * varphi_col) * h2R[None, :]
        return Field.from_circular(E_L, E_R, grid_out, symmetric=True)

    if field.symmetry == 'polar':
        h1r   = HT(1, r, tp * field.r,   q)
        h1phi = HT(1, r, ts * field.phi, q)
        E_r   = np.broadcast_to((2*π*h1r  )[None, :], (len(varphi), len(q))).copy()
        E_phi = np.broadcast_to((2*π*h1phi)[None, :], (len(varphi), len(q))).copy()
        return Field.from_polar(E_r, E_phi, grid_out, symmetric=True)

    if field.symmetry == 'cartesian':
        h0x = HT(0, r, (tp + ts) * field.x, q)
        h2x = HT(2, r, (tp - ts) * field.x, q)
        h0y = HT(0, r, (tp + ts) * field.y, q)
        h2y = HT(2, r, (tp - ts) * field.y, q)
        c2  = np.cos(2 * varphi_col)
        s2  = np.sin(2 * varphi_col)
        E_x = π * (h0x[None, :] - c2 * h2x[None, :] - s2 * h2y[None, :])
        E_y = π * (h0y[None, :] + c2 * h2y[None, :] - s2 * h2x[None, :])
        return Field.from_cartesian(E_x, E_y, grid_out, symmetric=True)

    if field.symmetry is None:
        raise NotImplementedError("Propagation requires a field with explicit symmetry.")
    raise ValueError(f"Unsupported field symmetry: {field.symmetry!r}")
