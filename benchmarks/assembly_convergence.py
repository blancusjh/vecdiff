"""Independent controls for a two-cap coherent spectral assembly.

These diagnostics expose model/spectral errors; they do not turn a converged
local-Fresnel feedback equation into an exact Maxwell boundary solution.
"""
import json
from pathlib import Path
import time
import numpy as np
from vecdiff import (Medium,Frame,SphericalCap,DielectricInterface,InterfaceAssembly,
    CartesianGrid,plane_wave,sample_surface,propagate_interfaces)
from vecdiff.observables.electromagnetism import boundary_residuals
from examples._report import versions,relative


def main():
    air,glass=Medium(),Medium(4.)
    caps=(SphericalCap(80.),SphericalCap(-80.,Frame(origin=[0,0,1.])))
    assembly=InterfaceAssembly((DielectricInterface(caps[0],air,glass),DielectricInterface(caps[1],glass,air)))
    controls=[('baseline',6,1.2,12),('bandwidth',8,.9,12),('bandwidth',10,.72,12),
              ('window',4,1.2,12),('window',8,1.2,12),('surface',6,1.2,24)]
    report=dict(versions=versions(),parameters=dict(wavelength=1.,indices=[1,4,1],radii=[80,-80],gap=1.,aperture=2.),
                scope='Finite periodic propagating lattice and local Fresnel curved traces. Boundary values below are continuation diagnostics inside source envelopes.',cases=[])
    destination=Path('benchmarks/results/assembly_convergence.json')
    baseline=None
    for name,count,spacing,nr in controls:
        start=time.perf_counter()
        sampling=tuple(sample_surface(s,(0,2.),(0,2*np.pi),nr,2*nr) for s in caps)
        f=propagate_interfaces(plane_wave(),assembly,CartesianGrid.from_spacing(spacing,count),samplings=sampling,rtol=1e-10)
        q=np.array([[-.5,0,2.],[0,0,2.],[.5,0,2.]])
        fields=np.concatenate(f.evaluate(q,region=2))
        if baseline is None:baseline=fields
        residuals=[]
        for j,surface in enumerate(caps):
            rho=np.linspace(.1,1.,11); points=surface.position(rho,.4)
            normal,_=surface.normal_and_jacobian(rho,.4)
            def trace(region):
                ef,hf=f.forward[region].evaluate(points);eb,hb=f.backward[region].evaluate(points)
                return ef+eb,hf+hb
            e1,h1=trace(j);e2,h2=trace(j+1)
            residuals.append(boundary_residuals(e1,h1,e2,h2,normal,assembly.media[j],assembly.media[j+1],electric_scale=1,magnetic_scale=1))
        row=dict(refinement=name,count=count,spacing=spacing,period=count*spacing,surface_radial_nodes=nr,
                 feedback_residual=f.feedback.relative_residual,iterations=f.feedback.iterations,
                 downstream_field_change=relative(fields,baseline),reconstructed_boundary=residuals,seconds=time.perf_counter()-start)
        report['cases'].append(row);destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
