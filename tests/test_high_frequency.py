import numpy as np
import pytest
from vecdiff import plane_wave,interface_transform,sample_surface,InterfaceAssembly,DielectricInterface,Medium,EvenAsphere,Frame
from examples.macroscopic_element import configuration,response
from vecdiff.propagation.high_frequency import propagate_high_frequency


def test_one_encounter_is_the_existing_per_mode_fresnel_radiation():
    assembly,a,focus=configuration();i=assembly.interfaces[0]
    q=sample_surface(i.surface,(0,a),(0,2*np.pi),12,24);source=plane_wave(wavelength=.03)
    fast=propagate_high_frequency(source,InterfaceAssembly((i,)),q)
    full=interface_transform(source,i,q)
    np.testing.assert_allclose(fast.transmitted.J,full.transmitted.J,atol=1e-12)
    np.testing.assert_allclose(fast.transmitted.M,full.transmitted.M,atol=1e-12)


def test_two_curved_faces_equal_optical_path_and_transverse_polarization():
    result,focus=response(nr=16,nphi=32)
    final=result.modes[0][-1]
    path=final.optical_path+np.linalg.norm(focus-final.sampling.points,axis=-1)
    assert np.ptp(path)<1e-10
    np.testing.assert_allclose(np.sum(final.transmitted_direction*final.boundary.transmitted_E,axis=-1),0,atol=1e-12)
    # Homogeneous transport preserves ray-tube power between the two faces.
    first=result.modes[0][0];second=result.modes[0][1]
    flux=lambda e,u,q: np.sum(abs(e)**2,axis=-1)*np.sum(u*q.normals,axis=-1)*q.weights
    np.testing.assert_allclose(flux(first.boundary.transmitted_E,first.transmitted_direction,first.sampling),
                               flux(second.boundary.incident_E,second.incident_direction,second.sampling),rtol=2e-12)


def test_intermediate_caustic_is_rejected():
    assembly,a,_=configuration();first=assembly.interfaces[0]
    plane=EvenAsphere(frame=Frame(origin=[0,0,35.]))
    chain=InterfaceAssembly((first,DielectricInterface(plane,Medium(1.5),Medium())))
    q=sample_surface(first.surface,(0,a),(0,2*np.pi),12,24)
    with pytest.raises(ValueError,match='caustic'):
        propagate_high_frequency(plane_wave(),chain,q)


def test_single_phase_and_plane_spectrum_are_identical_and_modes_remain_coherent():
    from vecdiff import EikonalElectricField,ElectricSpectrum
    assembly,a,focus=configuration();q=sample_surface(assembly.interfaces[0].surface,(0,a),(0,2*np.pi),12,24)
    source=plane_wave(wavelength=.03)
    phase=EikonalElectricField(lambda p:p[...,2],lambda p:[0,0,1],lambda p:[1,0,0],.03)
    ray=propagate_high_frequency(phase,assembly,q)
    spectral=propagate_high_frequency(source,assembly,q)
    np.testing.assert_allclose(ray.transmitted.J,spectral.transmitted.J,atol=1e-12)
    angle=.005;other=plane_wave((np.sin(angle),0,np.cos(angle)),(np.cos(angle),0,-np.sin(angle)),wavelength=.03)
    combined=ElectricSpectrum(np.concatenate((source.wavevectors,other.wavevectors)),
                              np.concatenate((source.amplitudes,1j*other.amplitudes)),.03)
    summed=propagate_high_frequency(combined,assembly,q)
    separate=propagate_high_frequency(other,assembly,q)
    points=focus+np.array([[-.1,0,0],[0,0,0],[.1,0,.1]])
    e,h=summed.transmitted.evaluate(points)
    e1,h1=spectral.transmitted.evaluate(points);e2,h2=separate.transmitted.evaluate(points)
    np.testing.assert_allclose(e,e1+1j*e2,rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(h,h1+1j*h2,rtol=2e-12,atol=2e-12)


def test_non_eikonal_phase_gradient_is_rejected():
    from vecdiff import EikonalElectricField
    assembly,a,_=configuration();q=sample_surface(assembly.interfaces[0].surface,(0,a),(0,2*np.pi),8,16)
    field=EikonalElectricField(lambda p:2*p[...,2],lambda p:[0,0,2],lambda p:[1,0,0])
    with pytest.raises(ValueError,match='phase gradient'):
        propagate_high_frequency(field,assembly,q)


def test_finite_conjugate_stigmatic_recovery():
    from examples.macroscopic_element import finite_conjugate_response
    result,focus=finite_conjugate_response(nr=12,nphi=24)
    end=result.modes[0][-1]
    opl=end.optical_path+np.linalg.norm(focus-end.sampling.points,axis=-1)
    np.testing.assert_allclose(opl,65.,atol=2e-11,rtol=0)
    np.testing.assert_allclose(end.transmitted_direction,(focus-end.sampling.points)/np.linalg.norm(focus-end.sampling.points,axis=-1)[:,None],atol=1e-12)
