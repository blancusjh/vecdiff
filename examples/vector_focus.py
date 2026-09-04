"""External Richards–Wolf reference using the native ElectricSpectrum state."""
import numpy as np
import matplotlib.pyplot as plt
from references.richards_wolf import spectrum
from ._report import main, relative


def run():
    """Compare pupil polarization and a vortex at equal incident pupil power."""
    x = np.linspace(-2, 2, 161)
    points = np.column_stack((x, 0*x, 0*x))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    rows = []
    for pol, charge in [("linear_x", 0), ("radial", 0), ("azimuthal", 0), ("linear_x", 1)]:
        fields = []
        for nt, np_ in [(24, 48), (48, 96)]:
            spec = spectrum(.9, polarization=pol, vortex_charge=charge, n_theta=nt, n_phi=np_)
            fields.append(spec.evaluate(points)[0])
        e = fields[-1]
        error = relative(e, fields[0])
        assert error < 1e-8
        label = f"{pol}, charge {charge}"
        axes[0].plot(x, np.sum(abs(e)**2, axis=-1), label=label)
        axes[1].plot(x, abs(e[:, 2])**2, label=label)
        rows.append(dict(polarization=pol, vortex_charge=charge, quadrature_change=error,
                         on_axis_components_norm2=(abs(e[len(x)//2])**2).tolist()))
    for ax, title in zip(axes, ["Total electric norm |E|²", "Longitudinal electric norm |Ez|²"]):
        ax.set(xlabel="x / wavelength", ylabel="Squared electric amplitude", title=title)
        ax.legend(fontsize=8)
    return fig, dict(parameters=dict(NA=.9, focal_length=1., wavelength=1.), cases=rows,
                    assumptions="Independent ideal sine-condition objective, uniform pupil magnitude and sqrt(cos θ) apodization. Not a dielectric lens or a Quabis experiment replication. Squared electric norm is not Poynting flux.")


if __name__ == "__main__":
    main("vector_focus", run)
