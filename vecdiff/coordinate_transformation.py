import numpy as np
exp = np.exp



# Polar - Cartesian 

def cartesian_to_polar(Ex, Ey, varphi):
    Er = Ex * np.cos(varphi) + Ey * np.sin(varphi)
    Ephi = -Ex * np.sin(varphi) + Ey * np.cos(varphi)
    return Er, Ephi


def polar_to_cartesian(Er, Ephi, varphi):
    Ex = Er * np.cos(varphi) - Ephi * np.sin(varphi)
    Ey = Er * np.sin(varphi) + Ephi * np.cos(varphi)
    return Ex, Ey


# Circular - Cartesian 

def circular_to_cartesian(EL, ER):
    Ex = (EL + ER) / np.sqrt(2)
    Ey = -1j * (EL - ER) / np.sqrt(2)
    return Ex, Ey


def cartesian_to_circular(Ex, Ey):
    EL = (Ex + 1j * Ey) / np.sqrt(2)
    ER = (Ex - 1j * Ey) / np.sqrt(2)
    return EL, ER


# Circular - Polar 

def circular_to_polar(EL, ER, varphi):
    Er = (exp(-1j * varphi) * EL + exp(1j * varphi) * ER) / np.sqrt(2)
    Ephi = (-1j * exp(-1j * varphi) * EL + 1j * exp(1j * varphi) * ER) / np.sqrt(2)
    return Er, Ephi


def polar_to_circular(Er, Ephi, varphi):
    EL = exp(1j * varphi) * (Er + 1j * Ephi) / np.sqrt(2)
    ER = exp(-1j * varphi) * (Er - 1j * Ephi) / np.sqrt(2)
    return EL, ER



def polar_grid_to_cartesian_grid(field_polar, q, varphi, x, y, fill_value=0.0):
    """
    Interpolate a field defined on a regular polar grid (varphi, q) to a Cartesian grid (x, y).

    Parameters
    ----------
    field_polar : array_like, shape (N_varphi, N_q)
        Complex or real samples on polar mesh.
    q : array_like, shape (N_q,)
        Radial grid (monotonic increasing, q >= 0).
    varphi : array_like, shape (N_varphi,)
        Angular grid in radians (monotonic increasing, periodic over 2*pi).
    x, y : array_like
        Cartesian coordinates. Must have same shape (typically meshgrid outputs).
    fill_value : float or complex
        Value used when rho is outside q range.

    Returns
    -------
    field_xy : ndarray, same shape as x
        Interpolated field on Cartesian grid.
    """
    F = np.asarray(field_polar)
    q = np.asarray(q, dtype=float)
    varphi = np.asarray(varphi, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if F.ndim != 2:
        raise ValueError("`field_polar` must be 2D with shape (N_varphi, N_q).")
    if q.ndim != 1 or varphi.ndim != 1:
        raise ValueError("`q` and `varphi` must be 1D arrays.")
    if F.shape != (varphi.size, q.size):
        raise ValueError("`field_polar` shape must be (len(varphi), len(q)).")
    if x.shape != y.shape:
        raise ValueError("`x` and `y` must have the same shape.")
    if q.size < 2 or varphi.size < 2:
        raise ValueError("`q` and `varphi` must have at least two samples.")

    rho = np.sqrt(x**2 + y**2)
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)

    # Angular interpolation (periodic)
    dphi = varphi[1] - varphi[0]
    tphi = np.mod((phi - varphi[0]) / dphi, varphi.size)
    i0 = np.floor(tphi).astype(int)
    i1 = (i0 + 1) % varphi.size
    wphi = tphi - i0

    # Radial interpolation (clamped inside domain)
    inside = (rho >= q[0]) & (rho <= q[-1])
    iq1 = np.searchsorted(q, rho, side="right")
    iq1 = np.clip(iq1, 1, q.size - 1)
    iq0 = iq1 - 1
    q0 = q[iq0]
    q1 = q[iq1]
    wq = np.where(q1 > q0, (rho - q0) / (q1 - q0), 0.0)

    # Bilinear interpolation
    f00 = F[i0, iq0]
    f01 = F[i0, iq1]
    f10 = F[i1, iq0]
    f11 = F[i1, iq1]

    f0 = (1.0 - wq) * f00 + wq * f01
    f1 = (1.0 - wq) * f10 + wq * f11
    out = (1.0 - wphi) * f0 + wphi * f1

    out = np.where(inside, out, fill_value)
    return out
