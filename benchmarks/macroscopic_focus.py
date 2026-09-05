"""Separate scale, surface quadrature, and local-expansion error controls."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from examples.macroscopic_focus import radiation, relative_eh, WAVELENGTH
from examples._report import versions


def run():
    cases=[]
    offsets=np.array([[x,y,z] for x in [-3,-1.5,0,1.5,3] for y in [-1,1] for z in [-12,-6,0,6,12]])
    for scale in (200.,2000.,20000.,10/WAVELENGTH):
        radius=scale*WAVELENGTH; center=np.array([0,0,2*radius]); points=center+WAVELENGTH*offsets
        start=perf_counter(); rad=radiation(radius); field=rad.local_spectrum(center,13*WAVELENGTH)
        build=perf_counter()-start
        start=perf_counter(); answer=field.evaluate(points); spectral_seconds=perf_counter()-start
        start=perf_counter(); reference=rad.evaluate(points); reference_seconds=perf_counter()-start
        radial=radiation(radius,nr=48,nphi=64).evaluate(points)
        azimuth=radiation(radius,nr=32,nphi=96).evaluate(points)
        combined=radiation(radius,nr=48,nphi=96).evaluate(points)
        case=dict(radius_over_wavelength=scale,radius_mm=radius,construction_seconds=build,
                  spectral_sample_seconds=spectral_seconds,reference_sample_seconds=reference_seconds,
                  relative_EH_kernel_error=relative_eh(answer,reference),
                  radial_quadrature_change=relative_eh(reference,radial),
                  azimuthal_quadrature_change=relative_eh(reference,azimuth),
                  combined_quadrature_change=relative_eh(reference,combined),
                  electric_absolute_bound=field.electric_error_bound,magnetic_absolute_bound=field.magnetic_error_bound)
        cases.append(case);print(case,flush=True)
    result=dict(versions=versions(),vacuum_wavelength_mm=WAVELENGTH,observation_ball_wavelengths=13,
                source_quadrature=[32,64],held_out_points=len(offsets),cases=cases,
                scope='Approximation error relative to the same prescribed-current radiation, not dielectric-boundary accuracy.')
    return result


if __name__=='__main__':
    Path('benchmarks/results/macroscopic_focus.json').write_text(json.dumps(run(),indent=2)+'\n')
