import numpy as np

π = np.pi
exp = np.exp

from .hankel import HankelTransform


def make_observation_grid(half_size, n_img):
    x = np.linspace(-half_size, half_size, n_img)
    y = np.linspace(-half_size, half_size, n_img)
    xx, yy = np.meshgrid(x, y)
    rho = np.sqrt(xx**2 + yy**2)
    varphi = np.arctan2(yy, xx)
    extent = [-half_size, half_size, -half_size, half_size]
    return xx, yy, rho, varphi, extent


def radial_to_2d(profile_q, q, rho):
    profile_q = np.asarray(profile_q)
    q = np.asarray(q)
    return np.interp(rho, q, profile_q, left=profile_q[0], right=profile_q[-1])


def hankel_terms(r, q, tp, ts, e1, e2, polarization):
    ht = HankelTransform.transform_array
    rep = str(polarization).lower()

    h0 = lambda f_r: ht(0, r, f_r, q)
    h1 = lambda f_r: ht(1, r, f_r, q)
    h2 = lambda f_r: ht(2, r, f_r, q)

    if rep == "circular":
        return {
            "E_LL": 2.0 * π * h0((tp + ts) * e1),
            "E_LR": -2.0 * π * h2((tp - ts) * e2),
            "E_RL": -2.0 * π * h2((tp - ts) * e1),
            "E_RR": 2.0 * π * h0((tp + ts) * e2),
        }

    if rep == "cartesian":
        return {
            "H0x": h0((tp + ts) * e1),
            "H0y": h0((tp + ts) * e2),
            "H2x": h2((tp - ts) * e1),
            "H2y": h2((tp - ts) * e2),
        }

    if rep == "polar":
        return {
            "Err": 2.0 * π * h1(tp * e1),
            "Ephiphi": 2.0 * π * h1(ts * e2),
        }

    raise ValueError("`polarization` must be one of: 'circular', 'polar', 'cartesian'.")


def reconstruct_2d_from_terms(terms, q, rho, varphi, polarization):
    rep = str(polarization).lower()

    if rep == "circular":
        e_ll = radial_to_2d(terms["E_LL"], q, rho)
        e_lr = radial_to_2d(terms["E_LR"], q, rho)
        e_rl = radial_to_2d(terms["E_RL"], q, rho)
        e_rr = radial_to_2d(terms["E_RR"], q, rho)
        e_l = e_ll + exp(2.0j * varphi) * e_lr
        e_r = exp(-2.0j * varphi) * e_rl + e_rr
        return e_l, e_r

    if rep == "cartesian":
        h0x = radial_to_2d(terms["H0x"], q, rho)
        h0y = radial_to_2d(terms["H0y"], q, rho)
        h2x = radial_to_2d(terms["H2x"], q, rho)
        h2y = radial_to_2d(terms["H2y"], q, rho)
        e_x = π * (h0x - np.cos(2.0 * varphi) * h2x - np.sin(2.0 * varphi) * h2y)
        e_y = π * (-np.sin(2.0 * varphi) * h2x + h0y + np.cos(2.0 * varphi) * h2y)
        return e_x, e_y

    if rep == "polar":
        e_r = radial_to_2d(terms["Err"], q, rho)
        e_phi = radial_to_2d(terms["Ephiphi"], q, rho)
        return e_r, e_phi

    raise ValueError("`polarization` must be one of: 'circular', 'polar', 'cartesian'.")
