import numpy as np
from .fresnel import FresnelOvoid
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
        return np.where(finite, t, np.interp(r, r[finite], t[finite]))
    return np.zeros_like(r, dtype=float)

def HT(order: int, r: np.ndarray, f_r: np.ndarray, q: np.ndarray) -> np.ndarray:
    return HankelTransform.transform_array(order, r, f_r, q)

def _build_output_grid(field: Field, q: np.ndarray, varphi: np.ndarray) -> Grid:
    """Build the polar output grid in q-space."""
    if field.grid.type == "polar":
        return Grid.from_polar(q, varphi)
    raise ValueError(f"Unsupported grid type: {type(field.grid)}")

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
        E_x = h0x[None, :] - c2 * h2x[None, :] - s2 * h2y[None, :]
        E_y = h0y[None, :] + c2 * h2y[None, :] - s2 * h2x[None, :]
        return Field.from_cartesian(E_x, E_y, grid_out, symmetric=True)

    if field.symmetry is None:
        raise NotImplementedError("Propagation requires a field with explicit symmetry.")
    raise ValueError(f"Unsupported field symmetry: {field.symmetry!r}")
