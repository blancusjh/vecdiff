"""Reproducible validation; run from the repository root.

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m benchmarks.validate_physics
Writes JSON with raw metrics, resolution and package versions. No fit, phase
alignment, intensity renormalization, or reference data enter native fields.
"""
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
from vecdiff import (Medium, Frame, Plane, SphericalCap, Sphere, DielectricInterface,
                     CartesianGrid, plane_wave, sample_surface, interface_transform)
from vecdiff.interfaces.fresnel import solve
from vecdiff.observables.electromagnetism import boundary_residuals, poynting
from vecdiff.propagation.surface_radiation import SurfaceRadiation
from references.mie import fields as mie_fields


def relative(a, b): return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))


def plane_sweep():
    rows = []
    angles = [0., 20., np.degrees(np.arcsin(1/1.5)), np.degrees(np.arctan(1.5)), 75., 89.9]
    q = np.stack((np.linspace(-2, 2, 23), np.zeros(23), np.zeros(23)), axis=-1)
    for n1, n2 in [(1., 1.5), (1.5, 1.), (1., 1.)]:
        for angle in angles:
            a = np.deg2rad(angle); u = np.array([np.sin(a), 0, np.cos(a)])
            for pol, e in [("s", [0, 1, 0]), ("p", [np.cos(a), 0, -np.sin(a)])]:
                wave = plane_wave(u, e, medium=Medium(n1))
                out = interface_transform(wave, DielectricInterface(Plane(), Medium(n1), Medium(n2)))
                ei, hi = wave.evaluate(q); er, hr = out.reflected.evaluate(q); et, ht = out.transmitted.evaluate(q)
                residual = boundary_residuals(ei+er, hi+hr, et, ht, [0, 0, 1], Medium(n1), Medium(n2), electric_scale=1, magnetic_scale=n1)
                flux = [float(np.mean(poynting(e, h)[:, 2])) for e, h in [(ei, hi), (er, hr), (et, ht)]]
                rows.append(dict(n1=n1, n2=n2, angle_deg=float(angle), polarization=pol,
                                 boundary=residual, flux_relative_error=abs((flux[0]+flux[1]-flux[2])/flux[0])))
    return rows


def cap_case(radius, level):
    """Boundary jumps of RECONSTRUCTED angular spectra at the same surface Q.

The tested model is an open spherical cap with aperture .6 R. Evaluate the
central .3 R to separate the observation region from the hard aperture edge.
Changing the numerical period does not change this physical aperture.
    """
    start = time.perf_counter(); aperture = .6*radius
    cap = SphericalCap(radius)
    nr = max(40, int(6*radius))*(level+1); nphi = 32*(level+1)
    sampling = sample_surface(cap, (0, aperture), (0, 2*np.pi), nr, nphi)
    wave = plane_wave(); interface = DielectricInterface(cap, Medium(), Medium(1.5))
    out = interface_transform(wave, interface, sampling)
    radial_count = max(512, int(32*radius))*(level+1)
    # Output azimuth resolves exp(i k rho cos(phi)) at the outer observation
    # radius: Nphi >= 2*k*rho plus margin. A constant Nphi fakes convergence
    # for macroscopic caps even when the surface-current harmonics are low.
    ntheta = max(64, int(10*radius))*(level+1)
    nphi_out = max(64, int(4*radius))*(level+1)
    r = out.reflected.angular_spectrum(direction=-1, n_theta=ntheta, n_phi=nphi_out,
                                       backend="polar", radial_count=radial_count, max_order=2)
    t = out.transmitted.angular_spectrum(direction=1, n_theta=ntheta, n_phi=nphi_out,
                                         backend="polar", radial_count=radial_count, max_order=2)
    rho = np.linspace(.01*radius, .3*radius, 17)
    q = np.concatenate((cap.position(rho, 0), cap.position(rho, np.pi/2)))
    normals = np.concatenate((cap.normal_and_jacobian(rho, 0)[0], cap.normal_and_jacobian(rho, np.pi/2)[0]))
    ei, hi = wave.evaluate(q); er, hr = r.evaluate(q); et, ht = t.evaluate(q)
    residual = boundary_residuals(ei+er, hi+hr, et, ht, normals, Medium(), Medium(1.5), electric_scale=1, magnetic_scale=1)
    b = out.boundary
    local = boundary_residuals(b.incident_E+b.reflected_E, b.incident_H+b.reflected_H,
                              b.transmitted_E, b.transmitted_H, sampling.normals, Medium(), Medium(1.5),
                              weights=sampling.weights, electric_scale=1, magnetic_scale=1)
    return dict(radius_over_wavelength=radius, level=level, aperture_over_radius=.6, observation_radius_over_radius=.3,
                nr=nr, nphi=nphi, radial_count=radial_count, angular_ntheta=ntheta, angular_nphi=nphi_out,
                propagating_spectrum_only=True, local_boundary=local, reconstructed_boundary=residual,
                seconds=time.perf_counter()-start), np.concatenate((er, hr, et, ht), axis=-1)


def sphere_case(radius, level):
    """Full closed-sphere diagnostic of an explicitly incomplete one-encounter model.

Interior boundary guess: local Fresnel transmission on illuminated hemisphere,
zero in shadow. Exterior scattered currents use (guessed total - incident)
on the ENTIRE sphere, so the test represents a closed geometry. It has no exit
encounter, repeated internal reflection/refraction, creeping waves, or resonant
feedback. This benchmark is NOT a production sphere solver. The discrepancy
with Mie is the measured consequence of applying the open-interface model
outside its domain, and is not attributed to an error in per-k Fresnel laws.
    """
    import miepython
    start = time.perf_counter(); sphere = Sphere(radius)
    # Split at the shadow discontinuity: integrate each hemisphere separately.
    from vecdiff import SurfaceSampling
    nmu = max(32, int(5*radius))*(level+1); nphi = max(64, int(12*radius))*(level+1)
    halves = [sample_surface(sphere, bounds, (0, 2*np.pi), nmu, nphi) for bounds in [(-1, 0), (0, 1)]]
    sampling = SurfaceSampling(sphere, np.concatenate([s.points for s in halves]),
                               np.concatenate([s.normals for s in halves]), np.concatenate([s.weights for s in halves]))
    wave = plane_wave(); interface = DielectricInterface(sphere, Medium(), Medium(1.5), normal_sign=-1)
    out = interface_transform(wave, interface, sampling, illuminated_only=True)
    b = out.boundary
    scattered = SurfaceRadiation.from_boundary(sampling, b.transmitted_E-b.incident_E, b.transmitted_H-b.incident_H,
                                               1., Medium(), normal_sign=1)
    # Bulk fields avoid singular boundary quadrature; actual boundary testing
    # is reported separately for the Mie reference and reconstructed cap.
    theta = np.linspace(.15, np.pi-.15, 13)
    directions = np.stack((np.sin(theta), np.zeros_like(theta), np.cos(theta)), axis=-1)
    outside, inside = directions*(1.3*radius), directions*(.7*radius)
    es, hs = scattered.evaluate(outside); ein, hin = wave.evaluate(outside)
    et, ht = out.transmitted.evaluate(inside)
    meo, mho = mie_fields(outside, radius); mei, mhi = mie_fields(inside, radius)
    # Check the independent reference boundary with a shrinking offset and
    # extra multipoles. Offset is an absolute fraction of vacuum wavelength.
    boundary = []
    for delta in (1e-5, 1e-7):
        eo, ho = mie_fields(directions*(radius+delta), radius)
        ei, hi = mie_fields(directions*(radius-delta), radius)
        boundary.append(dict(offset_over_wavelength=delta, **boundary_residuals(eo, ho, ei, hi, directions,
                            Medium(), Medium(1.5), electric_scale=1, magnetic_scale=1)))
    poles = int(np.ceil(2*np.pi*radius+4.05*(2*np.pi*radius)**(1/3)+2))+20
    more_e, more_h = mie_fields(np.concatenate((outside, inside)), radius, n_pole=poles)
    reference_convergence = relative(np.concatenate((meo, mei)), more_e)
    errors = dict(exterior_E=relative(ein+es, meo), exterior_H=relative(hin+hs, mho),
                  interior_E=relative(et, mei), interior_H=relative(ht, mhi))
    qext, qsca, qback, asymmetry = miepython.efficiencies(1.5, 2*radius, 1.)
    return dict(radius_over_wavelength=radius, size_parameter=2*np.pi*radius, level=level,
                nmu_per_hemisphere=nmu, nphi=nphi, one_encounter_errors_vs_mie=errors,
                mie_efficiencies=dict(extinction=float(qext), scattering=float(qsca), backscatter=float(qback), asymmetry=float(asymmetry)),
                mie_boundary=boundary,
                mie_extra_multipoles_E_change=reference_convergence,
                model="diagnostic only: one illuminated encounter, zero shadow trace, no repeated refraction or resonant feedback",
                seconds=time.perf_counter()-start), np.concatenate((ein+es, hin+hs, et, ht), axis=-1)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--radii", type=float, nargs="+", default=[.5, 1, 2, 5, 10, 50])
    parser.add_argument("--output", default="benchmarks/results/physics.json"); parser.add_argument("--skip-spheres", action="store_true")
    args = parser.parse_args()
    import scipy, miepython
    report = dict(versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, miepython=miepython.__version__),
                  conventions="exp(-i omega t), vacuum wavelength=1, H=Z0 H_SI, n1=1 n2=1.5", planes=plane_sweep(), caps=[], spheres=[])
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    def save(): output.write_text(json.dumps(report, indent=2)+"\n")
    save()
    for radius in args.radii:
        previous = None
        for level in (0, 1):
            row, fields = cap_case(radius, level)
            if previous is not None: row["quadrature_relative_field_change"] = relative(fields, previous)
            previous = fields; report["caps"].append(row); save(); print("cap", row, flush=True)
        if not args.skip_spheres:
            previous = None
            for level in (0, 1):
                row, fields = sphere_case(radius, level)
                if previous is not None: row["quadrature_relative_field_change"] = relative(fields, previous)
                previous = fields; report["spheres"].append(row); save(); print("sphere", row, flush=True)


if __name__ == "__main__": main()
