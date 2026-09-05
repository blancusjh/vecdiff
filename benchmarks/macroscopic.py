"""Actual spectral cap validation at hundreds of vacuum wavelengths.

No Mie or alternative boundary solver enters this calculation. Surface, polar,
and interpolation controls are refined separately. Analytic azimuthal synthesis
removes observation-phase aliasing without discarding current harmonics.
"""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from vecdiff import Medium,SphericalCap,DielectricInterface,plane_wave,sample_surface,interface_transform
from vecdiff.observables.electromagnetism import boundary_residuals
from examples._report import versions, relative


def power(rad,direction,n_theta,radial_count):
    """Integrated propagating power in Z0-scaled units, by spectral Parseval.

E coefficients include d²k/(2pi)² quadrature; divide |a|² by that measure.
This is not the sum of intensities of quadrature-weighted amplitudes.
    """
    n_phi=11
    s=rad.angular_spectrum(direction=direction,n_theta=n_theta,n_phi=n_phi,
                          backend='polar',radial_count=radial_count,max_order=2)
    theta,w=leggauss(n_theta); theta=(theta+1)*np.pi/4; w=w*np.pi/4
    k=rad.medium.wavenumber(rad.wavelength)
    measure=np.repeat(k*k*np.sin(theta)*np.cos(theta)*w*(2*np.pi/n_phi)/(2*np.pi)**2,n_phi)
    return float(.5*np.sum(np.sum(abs(s.amplitudes)**2,axis=-1)*abs(s.wavevectors[:,2].real)/(2*np.pi/rad.wavelength)/measure))


def case(radius, surface_factor=1, angular_factor=1, table_factor=1):
    start=time.perf_counter()
    cap=SphericalCap(radius); aperture=.6*radius
    nr=max(40,int(4*radius))*surface_factor
    nt=max(128,int(8*radius))*angular_factor
    table=max(512,int(16*radius))*table_factor
    sampling=sample_surface(cap,(0,aperture),(0,2*np.pi),nr,32)
    incident=plane_wave()
    out=interface_transform(incident,DielectricInterface(cap,Medium(),Medium(1.5)),sampling)
    rho=np.linspace(.01*radius,.3*radius,17)
    q=np.concatenate((cap.position(rho,0),cap.position(rho,np.pi/2)))
    normal=np.concatenate((cap.normal_and_jacobian(rho,0)[0],cap.normal_and_jacobian(rho,np.pi/2)[0]))
    obs=np.array([[-.1*radius,0,radius],[0,0,radius],[.1*radius,0,radius]])
    parameters=dict(n_theta=nt,radial_count=table,max_order=2)
    er,hr=out.reflected.evaluate_propagating(q,direction=-1,**parameters)
    et,ht=out.transmitted.evaluate_propagating(q,direction=1,**parameters)
    ei,hi=incident.evaluate(q)
    bulk=out.transmitted.evaluate_propagating(obs,direction=1,**parameters)
    residual=boundary_residuals(ei+er,hi+hr,et,ht,normal,Medium(),Medium(1.5),electric_scale=1,magnetic_scale=1)
    pin=.5*np.pi*aperture**2
    reflected=power(out.reflected,-1,nt,table)/pin
    transmitted=power(out.transmitted,1,nt,table)/pin
    row=dict(radius_over_wavelength=radius,aperture_over_radius=.6,radial_nodes=nr,azimuthal_nodes=32,
             polar_nodes=nt,interpolation_nodes=table,reconstructed_boundary=residual,
             reflected_power_fraction=reflected,transmitted_power_fraction=transmitted,
             power_balance_error=abs(reflected+transmitted-1),seconds=time.perf_counter()-start,
             process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    return row,np.concatenate((er,hr,et,ht)),np.concatenate(bulk)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--radii',nargs='+',type=float,default=[50,100,200])
    parser.add_argument('--output',type=Path,default=Path('benchmarks/results/macroscopic.json'))
    args=parser.parse_args()
    report=dict(versions=versions(),method='per-k Fresnel currents, propagating Fourier–Bessel spectrum',
        scope='Open caps: boundary continuation includes hard-aperture and omitted-evanescent effects; not full instrument validation.',
        memory='Linux ru_maxrss: cumulative process high-water mark, not isolated per-case memory.',cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for radius in args.radii:
        baseline=None
        for name,factors in [('baseline',(1,1,1)),('surface',(2,1,1)),('polar',(1,2,1)),('table',(1,1,2)),('combined',(2,2,2))]:
            row,trace,bulk=case(radius,*factors);row['refinement']=name
            if baseline is None: baseline=(trace,bulk)
            else:
                row['boundary_field_change']=relative(trace,baseline[0])
                row['downstream_field_change']=relative(bulk,baseline[1])
            report['cases'].append(row)
            args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(json.dumps(row),flush=True)


if __name__=='__main__': main()
