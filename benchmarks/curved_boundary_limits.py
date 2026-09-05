"""Two-sided full-Green boundary limits of the main curved-interface method.

No reference boundary data are used. These tests can FAIL physical acceptance
while passing numerical convergence: that distinction is the intended result.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from vecdiff import Medium,EvenAsphere,SphericalCap,DielectricInterface,plane_wave,interface_transform,ElectricSpectrum
from vecdiff.sampling.near_surface import sample_near_surface
from vecdiff.observables.electromagnetism import boundary_residuals
from examples._report import versions


def extrapolate(values):
    """Quadratic δ→0 extrapolation at δ,δ/2,δ/4; no fitted phase or amplitude."""
    a,b,c=values
    return a/3-2*b+8*c/3


def case(radius,n1,n2,*,angle=0.,flat=False,illumination="plane",probe_radius=.2):
    surface=EvenAsphere() if flat else SphericalCap(radius)
    aperture=.5*radius; rho=probe_radius*radius; phi=np.pi/4
    q=surface.position(rho,phi); slope=surface.slope(rho)
    # Analytic graph normal also covers the polar chart singularity at rho=0.
    normal=surface.frame.vectors(np.array([-slope*np.cos(phi),-slope*np.sin(phi),1.])/np.sqrt(1+slope*slope))
    theta=np.deg2rad(angle)
    incident=plane_wave((np.sin(theta),0,np.cos(theta)),(np.cos(theta),0,-np.sin(theta)),medium=Medium(n1))
    if illumination == 'gaussian':
        from numpy.polynomial.hermite import hermgauss
        nodes,weights=hermgauss(11); waist=aperture/2
        kx,ky=np.meshgrid(2*nodes/waist,2*nodes/waist); k1=2*np.pi*n1
        if np.any(kx*kx+ky*ky>=k1*k1): raise ValueError('Gaussian quadrature has nonpropagating modes')
        kz=np.sqrt(k1*k1-kx*kx-ky*ky); a=np.outer(weights,weights)/np.pi
        incident=ElectricSpectrum(np.stack((kx,ky,kz),axis=-1).reshape(-1,3),
            np.stack((a,0*a,-kx*a/kz),axis=-1).reshape(-1,3),medium=Medium(n1))
    elif illumination != 'plane': raise ValueError('unknown illumination')
    interface=DielectricInterface(surface,Medium(n1),Medium(n2))
    offsets=np.array([.008,.004,.002,.001])
    pair=[]; counts=[]; start=time.perf_counter()
    lower=q-offsets[:,None]*normal; upper=q+offsets[:,None]*normal
    ei,hi=incident.evaluate(lower)
    for order,azimuth in [(8,max(128,int(24*max(n1,n2)*rho))),(12,max(256,int(48*max(n1,n2)*rho)))]:
        # Hold each quadrature fixed through the offset sequence, resolving its smallest gap.
        samples=sample_near_surface(surface,aperture,[rho*np.cos(phi),rho*np.sin(phi)],offsets[-1],
            radial_panels=max(12,int(np.ceil(6*max(n1,n2)*aperture))),order=order,nphi=azimuth)
        out=interface_transform(incident,interface,samples)
        er,hr=out.reflected.evaluate(lower,chunk=1); et,ht=out.transmitted.evaluate(upper,chunk=1)
        pair.append(np.stack((ei+er,hi+hr,et,ht),axis=1)); counts.append(len(samples.points))
    changes=np.linalg.norm(pair[0]-pair[1],axis=(1,2))/np.linalg.norm(pair[1],axis=(1,2))
    traces=pair[1]
    coarse=extrapolate(traces[:3]); limit=extrapolate(traces[1:])
    limit_change=float(np.linalg.norm(limit-coarse)/np.linalg.norm(limit))
    def residual(trace):
        return boundary_residuals(*[a[None] for a in trace],normal,Medium(n1),Medium(n2),electric_scale=1,magnetic_scale=n1)
    jumps=residual(limit)
    local_scale=float(np.linalg.norm(incident.evaluate(q[None])[0]))
    return dict(radius_over_wavelength=radius,illumination=illumination,indices=[n1,n2],incidence_degrees=angle,flat_control=flat,
        aperture_over_radius=.5,probe_radius_over_radius=probe_radius,probe_azimuth_degrees=45,
        offsets_over_wavelength=offsets.tolist(),offset_boundary_residuals=[residual(t) for t in traces],
        extrapolated_boundary_residuals=jumps,incident_electric_norm_at_probe=local_scale,
        local_amplitude_normalized_boundary_residuals={key:value/local_scale for key,value in jumps.items()},max_quadrature_change=float(max(changes)),
        boundary_extrapolation_change=limit_change,max_source_nodes=max(counts),
        numerical_convergence_passed=bool(max(changes)<1e-6 and limit_change<1e-5),
        boundary_acceptance_at_one_percent=max(jumps.values())<.01,seconds=time.perf_counter()-start)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('benchmarks/results/curved_boundary_limits.json'))
    a=p.parse_args();report=dict(versions=versions(),method='Per-k Fresnel traces; full dyadic Green evaluation on both sides; target-centred quadrature; quadratic boundary-limit extrapolation.',
        scope='Necessary Maxwell continuity test of the supplied finite-aperture model. Hard-aperture edge and missing global boundary correction are included; this is not a closed-body solution.',cases=[])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    cases=[(r,n1,n2,0.,False,'plane') for r in [2.,10.,30.] for n1,n2 in [(1.,1.5),(1.5,1.)]]
    cases += [(10.,1.,1.,0.,False,'plane'),(10.,1.,1.5,0.,True,'plane'),(10.,1.,1.5,25.,False,'plane')]
    cases += [(10.,1.,n,0.,False,'gaussian') for n in [1.,1.5]]
    for r,n1,n2,angle,flat,illumination in cases:
        row=case(r,n1,n2,angle=angle,flat=flat,illumination=illumination);report['cases'].append(row)
        a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(row),flush=True)

    report['localized_probe_scan']=[]
    for rho in [0.,.1,.2,.3,.4]:
        for n2 in [1.,1.5]:
            row=case(10.,1.,n2,illumination='gaussian',probe_radius=rho)
            report['localized_probe_scan'].append(row)
            a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'); print(json.dumps(row),flush=True)


if __name__=='__main__':main()
