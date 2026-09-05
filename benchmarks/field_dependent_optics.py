"""Non-shift-invariant response and numerical controls for macroscopic surfaces."""
import json,time
from pathlib import Path
import numpy as np
from examples.field_dependent_optics import response,WAVELENGTH
from examples._report import versions


def case(kind,angle):
    start=time.perf_counter();rad,center=response(kind,angle)
    x=np.linspace(-15,15,241)*WAVELENGTH
    points=center+np.c_[x,0*x,0*x]
    result=rad.evaluate_local(points,radius=4*WAVELENGTH)
    perf=time.perf_counter()-start
    held=points[::12]
    full=rad.evaluate(held)
    tested=rad.evaluate_local(held,radius=4*WAVELENGTH)
    fine,_=response(kind,angle,nr=256,nphi=512)
    refined=fine.evaluate(held)
    def relative(a,b):return float(np.linalg.norm(np.concatenate(a,axis=-1)-np.concatenate(b,axis=-1))/np.linalg.norm(np.concatenate(b,axis=-1)))
    kernel=relative((tested.electric,tested.magnetic),full)
    quadrature=relative(full,refined)
    bound_passed=bool(np.max(np.linalg.norm(tested.electric-full[0],axis=-1))<=tested.electric_error_bound and np.max(np.linalg.norm(tested.magnetic-full[1],axis=-1))<=tested.magnetic_error_bound)
    intensity=np.sum(abs(result.electric)**2,axis=-1)
    return dict(kind=kind,angle_degrees=angle,center_mm=center.tolist(),source_nodes=len(rad.sampling.points),
        patch_count=result.patch_count,construction_and_line_seconds=perf,kernel_relative_EH_error=kernel,
        surface_quadrature_relative_EH_change=quadrature,absolute_kernel_bounds_passed=bound_passed,
        peak=float(intensity.max()),peak_offset_wavelengths=float(x[np.argmax(intensity)]/WAVELENGTH)),intensity


def main():
    report=dict(versions=versions(),wavelength_mm=WAVELENGTH,
        scope='Distinct per-k incident directions; actual refracting conic or reflecting paraboloid; fixed-source Green kernel checks. No translated PSF. No claim of global dielectric-boundary accuracy or complete multi-surface transport.',cases=[])
    for kind in ['refraction','reflection']:
        baseline=None
        for angle in [0.,.002,.01,.02]:
            row,intensity=case(kind,angle)
            if baseline is None:baseline=intensity
            row['peak_relative_to_on_axis']=float(intensity.max()/baseline.max())
            row['translated_on_axis_intensity_relative_error']=float(np.linalg.norm(intensity-baseline)/np.linalg.norm(intensity))
            report['cases'].append(row); print(json.dumps(row),flush=True)
    out=Path('benchmarks/results/field_dependent_optics.json');out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
