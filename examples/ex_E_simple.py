import numpy as np
from vecdiff.CartesianSurfaces import CartesianSurface
from vecdiff.fields import FieldCircular, FieldCartesian, FieldPolar


def main():
    # System and sampling
    lam = 532e-6
    n0 = 1.0
    ni = 1.5
    z0 = 5.0
    zi = 10.0
    r_max = 0.30
    n_samples = 400
    varphi = π / 8.0

    diopter = CartesianSurface(n0=n0, z0=z0, ni=ni, zi=zi)
    r = np.linspace(0.0, r_max, n_samples)
    q_max = ni / (lam * zi) * r_max
    q = np.linspace(0.0, q_max, n_samples)
    aperture = (r <= 0.22).astype(float).astype(complex)
    obs = {"r": r, "q": q, "varphi": varphi}

    # Minimal input fields (same aperture envelope)
    E_circ = FieldCircular(L=aperture, R=1j * aperture)
    E_cart = FieldCartesian(x=aperture, y=0.5 * aperture)
    E_polar = FieldPolar(r=aperture, phi=0.3 * aperture)

    # Propagation (currently implemented for z == diopter.zi)
    E_circ_p = E_circ.propagate(z=zi, diopter=diopter, observation=obs)
    E_cart_p = E_cart.propagate(z=zi, diopter=diopter, observation=obs)
    E_polar_p = E_polar.propagate(z=zi, diopter=diopter, observation=obs)

    print("Circular ->", E_circ_p.polarization, "| shapes:", E_circ_p.L.shape, E_circ_p.R.shape)
    print("Cartesian ->", E_cart_p.polarization, "| shapes:", E_cart_p.x.shape, E_cart_p.y.shape)
    print("Polar ->", E_polar_p.polarization, "| shapes:", E_polar_p.r.shape, E_polar_p.phi.shape)


if __name__ == "__main__":
    main()
