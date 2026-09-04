"""Evanescent coupling across a low-index gap between two prisms."""
import numpy as np
import matplotlib.pyplot as plt
from vecdiff import Medium, LayerStack, plane_wave, propagate_layers
from ._report import main


def run():
    """Show finite transmission through a gap above the single-interface critical angle."""
    angle = np.deg2rad(60.)
    gaps = np.geomspace(.002, 3., 150)
    fig, ax = plt.subplots(figsize=(6, 3.7), constrained_layout=True)
    max_error = 0.
    final = {}
    for pol, e in [("s", (0, 1, 0)), ("p", (np.cos(angle), 0, -np.sin(angle)))]:
        incident = plane_wave((np.sin(angle), 0, np.cos(angle)), e, medium=Medium(1.5))
        component = 0 if pol == "s" else 1
        values = []
        for gap in gaps:
            f = propagate_layers(incident, LayerStack((Medium(1.5), Medium(), Medium(1.5)), (gap,)))
            R = abs(f.reflection[0, component])**2
            T = abs(f.transmission[0, component])**2
            values.append(T)
            max_error = max(max_error, float(abs(R+T-1)))
        ax.semilogx(gaps, values, label=pol)
        final[pol] = dict(thinnest_T=float(values[0]), thickest_T=float(values[-1]))
    ax.set(xlabel="Gap / vacuum wavelength", ylabel="Power transmittance", title="Frustrated total internal reflection, 60°")
    ax.legend()
    assert max_error < 1e-12 and all(v["thinnest_T"] > .999 for v in final.values())
    return fig, dict(parameters=dict(indices=[1.5, 1., 1.5], angle_deg=60., gaps=gaps.tolist()),
                    assumptions="Lossless parallel media; exterior media identical, so T=|t|² here only.",
                    max_flux_error=max_error, limiting_values=final)


if __name__ == "__main__":
    main("frustrated_tir", run)
