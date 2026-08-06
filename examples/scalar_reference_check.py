"""Which pupil weighting solves the *scalar* refraction problem.

The scalar transmission problem through a dielectric interface is the
Helmholtz equation in both media with ``U`` and ``dU/dn`` continuous on the
surface.  That continuity gives, mode by mode, the transmission coefficient
``2 n0 cos(theta_0) / (n0 cos(theta_0) + ni cos(theta_i))`` -- exactly the
electromagnetic ``t_s``; the scalar problem is the TE problem.  Flux transport
in a ray tube is polarization-blind, so the geometric factor ``A(Q)`` and the
sine mapping are unchanged.  The right scalar version of the stigmatic diopter
is therefore the s channel of the transfer operator, and nothing else.

This script puts that claim on trial against the exact scalar field: the
Helmholtz-Kirchhoff integral over the real surface with physical-optics
boundary data (:func:`vecdiff.reference.scalar_focal_field_reference`).  Three
candidates are compared:

* the s channel ``A t_s jac`` -- the claim;
* the polarization-averaged weight ``(w_p + w_s)/2`` -- carries the meridional
  projection, a vectorial object with no place in a scalar problem;
* the flat pupil ``t = 1`` -- the ideal-lens reference, which never solved an
  interface.

Run from the repository root::

    uv run python examples/scalar_reference_check.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from scipy.special import jv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vecdiff import CartesianSurface  # noqa: E402
from vecdiff.pupil_mapping import debye_prefactor  # noqa: E402
from vecdiff.reference import scalar_focal_field_reference  # noqa: E402
from vecdiff.transfer import focal_channel_weights  # noqa: E402


def main():
    lam = 193e-6
    surface = CartesianSurface(n0=2.14, ni=1.437, z0=-42.0, zi=2.0)
    aperture = 0.999 * surface.aperture_limit
    k = 2.0 * np.pi * surface.ni / lam

    edge = surface.ray_geometry(np.array([aperture]))
    print(f"superficie: LuAG → agua, NA = {surface.ni * edge.sin_ai[0]:.4f}")

    rho = np.linspace(0.0, 3.0, 101) * lam
    obs = np.stack([rho, np.zeros_like(rho), np.full_like(rho, float(surface.zi))], axis=-1)

    print("integrando la referencia escalar de Helmholtz-Kirchhoff ...")
    U_ref = scalar_focal_field_reference(
        surface, lam, lambda r: np.ones_like(r),
        aperture=aperture, observation=obs, n_r=2600, n_phi=224,
    )

    r = np.linspace(0.0, aperture, 6001)
    w_p, w_s, _w_z, u = focal_channel_weights(surface, r)
    q = k * rho / abs(surface.zi)

    def model(w):
        integrand = w[None, :] * jv(0, q[:, None] * u[None, :]) * u[None, :]
        return debye_prefactor(surface, lam) * simpson(integrand, x=u, axis=1)

    candidates = {
        "canal s  (A·t_s·jac)": w_s,
        "promedio (w_p+w_s)/2": 0.5 * (w_p + w_s),
        "pupila plana t=1": np.ones_like(u),
    }

    profile_ref = np.abs(U_ref) / np.abs(U_ref).max()
    print()
    print(f"{'candidato':<24} {'|U(0)|/ref':>12} {'fase(0)':>9} {'RMS perfil':>12}")
    print("-" * 62)
    for name, w in candidates.items():
        U = model(w)
        ratio = U[0] / U_ref[0]
        profile = np.abs(U) / np.abs(U).max()
        rms = float(np.sqrt(np.mean((profile - profile_ref) ** 2)))
        print(f"{name:<24} {abs(ratio):>12.6f} {np.degrees(np.angle(ratio)):>8.3f}° {rms:>12.2e}")


if __name__ == "__main__":
    main()
