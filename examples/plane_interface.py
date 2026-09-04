"""Brewster reflection and total internal reflection from the per-k interface map."""
import numpy as np
import matplotlib.pyplot as plt
from vecdiff import Medium, Plane, DielectricInterface, plane_wave, interface_transform
from vecdiff.observables.electromagnetism import boundary_residuals, poynting
from ._report import main


def run():
    """Test all four boundary conditions after evaluating both output spectra."""
    angles = np.linspace(0, 88, 177)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    rows = []
    q = np.array([[.13, .21, 0], [-.47, .31, 0]])
    for ax, (n1, n2) in zip(axes, [(1., 1.5), (1.5, 1.)]):
        curves = {"s": [], "p": []}
        for angle in angles:
            theta = np.deg2rad(angle)
            for pol, e in [("s", (0, 1, 0)), ("p", (np.cos(theta), 0, -np.sin(theta)))]:
                incident = plane_wave((np.sin(theta), 0, np.cos(theta)), e, medium=Medium(n1))
                result = interface_transform(incident, DielectricInterface(Plane(), Medium(n1), Medium(n2)))
                ei, hi = incident.evaluate(q)
                er, hr = result.reflected.evaluate(q)
                et, ht = result.transmitted.evaluate(q)
                jumps = boundary_residuals(ei+er, hi+hr, et, ht, [0, 0, 1], Medium(n1), Medium(n2),
                                           electric_scale=1, magnetic_scale=n1)
                flux = [float(np.mean(poynting(a, b)[:, 2])) for a, b in ((ei, hi), (er, hr), (et, ht))]
                R, T = -flux[1]/flux[0], flux[2]/flux[0]
                curves[pol].append(R)
                rows.append(dict(n1=n1, n2=n2, angle=float(angle), polarization=pol,
                                 R=R, T=T, max_boundary_residual=max(jumps.values())))
        for pol, curve in curves.items():
            ax.plot(angles, curve, label=pol)
        ax.set(xlabel="Incidence angle / degrees", ylabel="Power reflectance", title=f"n={n1} → {n2}", ylim=(-.02, 1.02))
        ax.legend()
    boundary = max(r["max_boundary_residual"] for r in rows)
    flux = max(abs(r["R"]+r["T"]-1) for r in rows)
    assert boundary < 1e-12 and flux < 1e-12
    return fig, dict(parameters=dict(wavelength=1., angles=angles.tolist()),
                     assumptions="Infinite planes; lossless media; unit incident electric amplitude.",
                     max_boundary_residual=boundary, max_flux_error=flux, cases=rows)


if __name__ == "__main__":
    main("plane_interface", run)
