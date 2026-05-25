import numpy as np

from .fresnel import FresnelOvoid
from .hankel import HankelTransform
from .fields import FieldCircular, FieldCartesian, FieldPolar



exp = np.exp
π = np.pi


def _get_observation_arrays(observation):
    if not isinstance(observation, dict):
        raise ValueError("`observation` must be a dict with keys: 'r', 'q', and optional 'varphi'.")
    if "r" not in observation or "q" not in observation:
        raise ValueError("`observation` must include 'r' and 'q'.")
    r = np.asarray(observation["r"], dtype=float)
    q = np.asarray(observation["q"], dtype=float)
    varphi = float(observation.get("varphi", 0.0))
    return r, q, varphi


def HT(order, r, f_r, q):
    return HankelTransform.transform_array(order, r, f_r, q)


def propagate_to_focal_plane_through_diopter(field, diopter, observation):
    r, q, varphi = _get_observation_arrays(observation)
    fresnel = FresnelOvoid(ovoid=diopter)
    tp = fresnel.tp(r)
    ts = fresnel.ts(r)

  

    if isinstance(field, FieldCircular):

        h0L = HT(0, r, (tp + ts) * field.L, q)
        h2L = HT(2, r, (tp - ts)*field.R, q)

        h0R = HT(0, r, (tp + ts) * field.R, q) 
        h2R = HT(2, r, (tp - ts) * field.L, q)

        E_LL = 2*π*h0L
        E_LR = -2*π*exp(2.0j * varphi)*h2L

        E_RL = -2*π*exp(-2.0j * varphi)*h2R
        E_RR = 2*π*h0R

        E_L = E_LL + E_LR
        E_R = E_RL + E_RR
        
        return FieldCircular(E_L, E_R)

    if isinstance(field, FieldCartesian):
        h0x = HT(0, r, (tp + ts) * field.x, q)
        h2x = HT(2, r, (tp - ts) * field.x, q)
        h0y = HT(0, r, (tp + ts) * field.y, q)
        h2y = HT(2, r, (tp - ts) * field.y, q)

        Exx = h0x - np.cos(2*varphi) * h2x
        Exy = -np.sin(2*varphi) * h2y
        Eyx = -np.sin(2*varphi) * h2x
        Eyy = h0y + np.cos(2*varphi) * h2y
        E_x = Exx + Exy
        E_y = Eyx + Eyy
        return FieldCartesian(E_x, E_y)

    if isinstance(field, FieldPolar):
        h1r = HT(1, r, tp * field.r, q)
        h1phi = HT(1, r, ts * field.phi, q)

        Err = 2*π*h1r
        Ephiphi = 2*π*h1phi
        E_r = Err
        E_phi = Ephiphi
        return FieldPolar(E_r, E_phi)

    raise ValueError("Unsupported field type.")
