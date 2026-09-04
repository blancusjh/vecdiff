"""A finite spherical cap: numerical convergence is not boundary accuracy."""
import numpy as np
import matplotlib.pyplot as plt
from vecdiff import (Medium, SphericalCap, DielectricInterface, plane_wave,
                     sample_surface, interface_transform)
from vecdiff.observables.electromagnetism import boundary_residuals
from ._report import main, relative


def run():
    """Compare source quadratures and report reconstructed boundary jumps separately."""
    cap = SphericalCap(radius=20.)
    n1, n2 = Medium(), Medium(1.5)
    incident = plane_wave()
    x = np.linspace(-4, 4, 161)
    points = np.column_stack((x, 0*x, 0*x+60))
    fields = []
    for nr, nv in [(64, 48), (128, 96)]:
        sampling = sample_surface(cap, (0, 8), (0, 2*np.pi), nr, nv)
        result = interface_transform(incident, DielectricInterface(cap, n1, n2), sampling)
        fields.append(result.transmitted.evaluate(points)[0])
    b = result.boundary
    local = boundary_residuals(b.incident_E+b.reflected_E, b.incident_H+b.reflected_H,
                               b.transmitted_E, b.transmitted_H, sampling.normals, n1, n2,
                               weights=sampling.weights, electric_scale=1., magnetic_scale=1.)
    # This propagating-only continuation is intentionally a diagnostic, not a
    # singular on-surface Green quadrature or an exact Maxwell boundary solve.
    spectra = [rad.angular_spectrum(direction=sign, n_theta=240, n_phi=128,
                                    backend="polar", radial_count=1600, max_order=2)
               for rad, sign in [(result.reflected, -1), (result.transmitted, 1)]]
    rho = np.linspace(.1, 4, 31)
    q = cap.position(rho, .37)
    normals, _ = cap.normal_and_jacobian(rho, .37)
    ei, hi = incident.evaluate(q)
    er, hr = spectra[0].evaluate(q)
    et, ht = spectra[1].evaluate(q)
    reconstructed = boundary_residuals(ei+er, hi+hr, et, ht, normals, n1, n2,
                                       electric_scale=1., magnetic_scale=1.)
    change = relative(fields[1], fields[0])
    assert change < 1e-7 and max(local.values()) < 1e-12
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for e, label in zip(fields, ["64 × 48 nodes", "128 × 96 nodes"]):
        axes[0].plot(x, np.sum(abs(e)**2, axis=-1), label=label)
    axes[0].set(xlabel="x / wavelength", ylabel="|E|²", title="Transmitted field at z=60 λ")
    axes[0].legend()
    positions = np.arange(4)
    axes[1].bar(positions-.18, np.maximum(list(local.values()), 1e-17), .36, label="Prescribed trace")
    axes[1].bar(positions+.18, list(reconstructed.values()), .36, label="Reconstructed diagnostic")
    axes[1].set(yscale="log", xticks=positions, xticklabels=["Et", "Ht", "Dn", "Bn"], ylabel="Normalized boundary jump")
    axes[1].legend(fontsize=8)
    return fig, dict(parameters=dict(radius=20., aperture=8., wavelength=1., observation_z=60.),
                     assumptions="Local tangent-plane physical optics on an open cap. Reconstruction diagnostic also omits evanescent modes and includes a hard aperture; its error cannot be attributed to curvature alone.",
                     quadrature_relative_change=change, local_boundary=local, reconstructed_boundary=reconstructed)


if __name__ == "__main__":
    main("curved_interface", run)
