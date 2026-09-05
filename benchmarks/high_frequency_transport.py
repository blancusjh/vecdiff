"""Accuracy and cost of mode-wise eikonal transport between curved interfaces."""
import json,time
from pathlib import Path
import numpy as np
from numpy.polynomial.hermite import hermgauss
from vecdiff import ElectricSpectrum,sample_surface,interface_transform
from vecdiff.propagation.high_frequency import sample_high_frequency
from vecdiff.fields.eikonal_field import EikonalElectricField
from examples.macroscopic_element import configuration,response
from examples._report import versions


def gaussian_spectrum(wavelength,waist=1.,order=25):
    nodes,weights=hermgauss(order);kx,ky=np.meshgrid(2*nodes/waist,2*nodes/waist)
    k=2*np.pi/wavelength;kz=np.sqrt(k*k-kx*kx-ky*ky);a=np.outer(weights,weights)/np.pi
    return ElectricSpectrum(np.stack((kx,ky,kz),axis=-1).reshape(-1,3),
                            np.stack((a,0*a,-kx*a/kz),axis=-1).reshape(-1,3),wavelength)


def relative(a,b):
    return float(np.linalg.norm(np.concatenate(a,axis=-1)-np.concatenate(b,axis=-1))/np.linalg.norm(np.concatenate(b,axis=-1)))


def propagation_case(radius_over_wavelength):
    assembly,aperture,focus=configuration();wavelength=10/radius_over_wavelength
    source=gaussian_spectrum(wavelength)
    xy=np.array([[0,0],[.25,0],[.5,0],[.25,.25],[0,.5]])
    from vecdiff.propagation.high_frequency import _graph
    points,_=_graph(assembly.interfaces[-1].surface,xy)
    start=time.perf_counter();fast=sample_high_frequency(source,assembly,xy,derivative_step=1e-4,initial_aperture=aperture)
    fast_seconds=time.perf_counter()-start
    half=sample_high_frequency(source,assembly,xy,derivative_step=5e-5,initial_aperture=aperture)
    single=EikonalElectricField(lambda p:p[...,2],lambda p:np.array([0.,0.,1.]),
        lambda p:np.exp(-np.sum(p[...,:2]**2,axis=-1))[...,None]*np.array([1.,0.,0.]),wavelength)
    start=time.perf_counter();single_field=sample_high_frequency(single,assembly,xy,derivative_step=1e-4,initial_aperture=aperture)
    single_seconds=time.perf_counter()-start
    edge_phi=np.linspace(0,2*np.pi,128,endpoint=False)
    edge=np.c_[aperture*np.cos(edge_phi),aperture*np.sin(edge_phi),np.zeros(128)]
    input_edge=float(np.max(np.linalg.norm(source.evaluate(edge)[0],axis=-1)))
    direct=[];times=[];counts=[]
    # Resolve the complete oscillatory surface integral, independently of ray transport.
    nr=max(96,int(radius_over_wavelength*.25));nphi=max(96,int(radius_over_wavelength*.3))
    for radial,azimuth in [(nr,nphi),(int(nr*1.5),int(nphi*1.5))]:
        start=time.perf_counter()
        sampling=sample_surface(assembly.interfaces[0].surface,(0,aperture),(0,2*np.pi),radial,azimuth)
        radiation=interface_transform(source,assembly.interfaces[0],sampling).transmitted
        direct.append(radiation.evaluate(points,chunk=1));times.append(time.perf_counter()-start);counts.append(len(sampling.points))
    return dict(radius_over_wavelength=radius_over_wavelength,wavelength_mm=wavelength,incident_modes=len(source.amplitudes),
                observation_count=len(xy),source_nodes=counts,direct_seconds=times,transport_seconds=fast_seconds,
                complex_EH_relative_error=relative(fast,direct[-1]),single_phase_relative_error=relative(single_field,direct[-1]),
                single_phase_seconds=single_seconds,single_phase_speedup=times[-1]/single_seconds,input_edge_amplitude=input_edge,
                direct_quadrature_change=relative(direct[0],direct[1]),
                derivative_step_change=relative(fast,half),speedup_vs_fine_direct=times[-1]/fast_seconds,
                reference_converged=relative(direct[0],direct[1])<1e-5)


def system_case(wavelength,angle):
    start=time.perf_counter();result,focus=response(angle,wavelength=wavelength);transport=time.perf_counter()-start
    last=result.modes[0][-1]
    predicted=focus+np.array([45*np.tan(np.deg2rad(angle)),0,0])
    x=np.linspace(-8,8,201)*wavelength;points=predicted+np.c_[x,0*x,0*x]
    start=time.perf_counter();field=result.transmitted.evaluate_local(points,radius=3*wavelength);evaluation=time.perf_counter()-start
    refined,_=response(angle,wavelength=wavelength,nr=96,nphi=192,derivative_step=result.derivative_step/2)
    held=points[::20];full=result.transmitted.evaluate(held);fine=refined.transmitted.evaluate(held)
    opl=last.optical_path+np.linalg.norm(focus-last.sampling.points,axis=-1)
    intensity=np.sum(abs(field.electric)**2,axis=-1)
    return dict(wavelength_mm=wavelength,angle_degrees=angle,surface_samples=len(last.sampling.points),interfaces=2,
                transport_seconds=transport,line_evaluation_seconds=evaluation,source_quadrature_and_derivative_change=relative(full,fine),
                on_axis_optical_path_spread_wavelengths=float(np.ptp(opl)/wavelength) if angle==0 else None,
                peak=float(intensity.max()),peak_offset_wavelengths=float(x[np.argmax(intensity)]/wavelength))



def compression_case():
    from vecdiff.propagation.high_frequency import propagate_high_frequency
    assembly,aperture,focus=configuration(); wavelength=.000193368
    source=gaussian_spectrum(wavelength)
    single=EikonalElectricField(lambda p:p[...,2],lambda p:np.array([0.,0.,1.]),
        lambda p:np.exp(-np.sum(p[...,:2]**2,axis=-1))[...,None]*np.array([1.,0.,0.]),wavelength)
    points=focus+wavelength*np.c_[np.linspace(-16,16,17),np.zeros(17),np.zeros(17)]
    values=[];times=[]
    for nr,nphi in [(24,48),(32,64)]:
        sampling=sample_surface(assembly.interfaces[0].surface,(0,aperture),(0,2*np.pi),nr,nphi)
        start=time.perf_counter();result=propagate_high_frequency(source,assembly,sampling)
        field=result.transmitted.evaluate(points,chunk=1);times.append(time.perf_counter()-start);values.append(field)
        del result
    start=time.perf_counter();one=propagate_high_frequency(single,assembly,sampling)
    one_field=one.transmitted.evaluate(points,chunk=1);one_seconds=time.perf_counter()-start
    return dict(wavelength_mm=wavelength,radius_over_wavelength=10/wavelength,waist_mm=1.,interfaces=2,
        spectral_modes=len(source.amplitudes),single_phase_samples=32*64,multiple_phase_samples=[625*24*48,625*32*64],
        observation_count=len(points),multiple_phase_seconds=times,single_phase_seconds=one_seconds,
        single_vs_multiple_phase_relative_EH_error=relative(one_field,values[-1]),
        multiple_phase_source_quadrature_change=relative(values[0],values[1]),speedup=times[-1]/one_seconds,
        scope='Final two-interface image-field comparison of one eikonal versus 625 separately transported phase components. Both use the high-frequency inter-surface approximation; final radiation uses the full Green kernel.')


def main():
    path=Path('benchmarks/results/high_frequency_transport.json')
    report=dict(versions=versions(),method='One ray per initial sample and incident spectral mode; vector Fresnel transformations; ray-tube spreading; final current radiation.',
        scope='Explicit leading high-frequency approximation between interfaces. Direct full-Green controls use an independently defined 625-mode beam. No shift invariance. No intermediate edge diffraction, caustic crossing or resonant feedback.',propagation=[],systems=[])
    for size in [100,200,500,1000]:
        row=propagation_case(size);report['propagation'].append(row);print(json.dumps(row),flush=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for wavelength in [.1,.01,.000193368]:
        for angle in [0.,.002,.02]:
            row=system_case(wavelength,angle);report['systems'].append(row);print(json.dumps(row),flush=True)
            path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

    report['macroscopic_compression']=compression_case()
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
