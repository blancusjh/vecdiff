"""A closed-surface response with independent Mie and source-order checks."""
import numpy as np
import matplotlib.pyplot as plt
from benchmarks.closed_sphere import case
from references.mie import fields
from ._report import main


def run():
    """Resolve a dielectric sphere's wavelength response without using Mie in the solve."""
    wavelengths = np.linspace(.8, 1.2, 9)
    rows = []
    for wavelength in wavelengths:
        row, _, _ = case(.5, 12, wavelength=float(wavelength), sphere_index=2.)
        e, _ = fields([[0, 0, 0]], .5, wavelength=wavelength, sphere_index=2.)
        row["mie_center_electric_norm2"] = float(np.sum(abs(e)**2))
        rows.append(row)
    # Refine the sample with the largest measured Mie error, not the easiest one.
    worst = max(rows, key=lambda r: max(r["errors_vs_mie"].values()))
    refined, _, _ = case(.5, 16, wavelength=worst["wavelength"], sphere_index=2.)
    max_error = max(max(r["errors_vs_mie"].values()) for r in rows)
    assert max_error < 5e-3
    assert max(refined["errors_vs_mie"].values()) < max(worst["errors_vs_mie"].values())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    # A dense reference curve makes clear that the native wavelength scan is sparse.
    fine = np.linspace(.8, 1.2, 241)
    reference = [float(np.sum(abs(fields([[0, 0, 0]], .5, wavelength=w, sphere_index=2.)[0])**2)) for w in fine]
    axes[0].plot(fine, reference, label="Independent Mie")
    axes[0].plot(wavelengths, [r["center_electric_norm2"] for r in rows], "o", label="Maxwell boundary match")
    axes[0].set(xlabel="Vacuum wavelength / sphere diameter", ylabel="Center |E|²", title="Self-consistent internal field")
    axes[0].legend(fontsize=8)
    axes[1].semilogy(wavelengths, [max(r["errors_vs_mie"].values()) for r in rows], "o-", label="Maximum bulk E/H error")
    axes[1].semilogy(wavelengths, [max(r["held_out_boundary"].values()) for r in rows], "s-", label="Maximum held-out boundary jump")
    axes[1].set(xlabel="Vacuum wavelength / sphere diameter", ylabel="Relative error")
    axes[1].legend(fontsize=8)
    return fig, dict(parameters=dict(radius=.5, index=2., order=12, wavelength_count=9),
                    assumptions="Dense auxiliary-source boundary match, experimental. All interactions are implicit in the complex solve; this is not repeated local Fresnel tracing. Not a high-Q peak-resolution certificate; refine wavelength sampling for spectroscopy.",
                    max_bulk_error=max_error, cases=rows, worst_case_refinement=refined)


if __name__ == "__main__":
    main("sphere_resonance", run)
