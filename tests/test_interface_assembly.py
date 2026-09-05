import numpy as np
import pytest
from vecdiff import (Medium, Plane, Frame, SphericalCap, CartesianGrid,
    DielectricInterface, InterfaceAssembly, ElectricSpectrum, plane_wave,
    LayerStack, propagate_layers, propagate_interfaces, sample_surface, ConvergenceError)
from vecdiff.observables.electromagnetism import boundary_residuals


@pytest.mark.parametrize('indices', [(1., 1.5, 1.), (1., 4., 1.), (1.2, 2., 1.6, 1.)])
@pytest.mark.parametrize('method', ['successive', 'gmres'])
def test_constructed_encounters_match_independent_layer_recursion(indices, method):
    media = tuple(Medium(n) for n in indices)
    heights = np.arange(len(indices)-1)*1.37
    interfaces = tuple(DielectricInterface(Plane(Frame(origin=[0,0,z])), a,b)
                       for z,a,b in zip(heights,media,media[1:]))
    grid = CartesianGrid.from_spacing(1.,4)
    kx=np.array([0,grid.kxy[0][0,1]]); kz=np.sqrt((2*np.pi*indices[0])**2-kx**2)
    k=np.column_stack((kx,np.zeros(2),kz))
    a=np.array([[1,.3j,0],[.2,.1j,0]],complex); a[:,2]=-kx*a[:,0]/kz
    incident=ElectricSpectrum(k,a,medium=media[0])
    field=propagate_interfaces(incident,InterfaceAssembly(interfaces),grid,method=method,rtol=1e-11)
    exact=propagate_layers(incident,LayerStack(media,tuple(np.diff(heights))))
    for region in range(len(media)):
        z=-.4 if region==0 else (heights[-1]+.4 if region==len(media)-1 else (heights[region-1]+heights[region])/2)
        q=np.array([[.1,.2,z],[-.3,.1,z]])
        for actual,reference in zip(field.evaluate(q,region=region),exact.evaluate(q,region=region)):
            np.testing.assert_allclose(actual,reference,atol=4e-10,rtol=4e-10)
    for j,z in enumerate(heights):
        q=np.array([[.1,.2,z],[-.3,.1,z]])
        e1,h1=field.evaluate(q,region=j); e2,h2=field.evaluate(q,region=j+1)
        residual=boundary_residuals(e1,h1,e2,h2,[0,0,1],media[j],media[j+1],electric_scale=1,magnetic_scale=1)
        assert max(residual.values()) < 5e-10
    assert field.feedback.relative_residual < 1e-11


def test_curved_encounters_are_coherent_and_reject_invalid_continuation():
    air,glass=Medium(),Medium(1.5)
    front=SphericalCap(30.)
    back=SphericalCap(-30.,Frame(origin=[0,0,3.]))
    assembly=InterfaceAssembly((DielectricInterface(front,air,glass),DielectricInterface(back,glass,air)))
    samples=tuple(sample_surface(s,(0,1.5),(0,2*np.pi),12,20) for s in (front,back))
    grid=CartesianGrid.from_spacing(1.1,4)
    args=dict(samplings=samples,rtol=1e-8)
    f=propagate_interfaces(plane_wave(),assembly,grid,**args)
    g=propagate_interfaces(plane_wave(polarization=(1j,0,0)),assembly,grid,**args)
    for region,z in enumerate([-1,1.5,4]):
        for a,b in zip(g.evaluate([[0,0,z]],region=region),f.evaluate([[0,0,z]],region=region)):
            np.testing.assert_allclose(a,1j*b,atol=2e-10)
    with pytest.raises(ValueError,match='source-free spectral slab'):
        f.evaluate([[0,0,0]],region=1)
    assert f.feedback.iterations > 1


def test_assembly_rejects_unsupported_geometry_and_spectral_support():
    air,glass=Medium(),Medium(1.5)
    a=DielectricInterface(Plane(),air,glass)
    b=DielectricInterface(Plane(Frame(origin=[0,0,2.])),glass,air)
    assembly=InterfaceAssembly((a,b))
    with pytest.raises(ValueError,match='grazing/evanescent'):
        propagate_interfaces(plane_wave(),assembly,CartesianGrid.from_spacing(.2,4))
    with pytest.raises(ValueError,match='same medium'):
        InterfaceAssembly((a,a))
    with pytest.raises(ValueError,match='separated and ordered'):
        propagate_interfaces(plane_wave(),InterfaceAssembly((a,DielectricInterface(Plane(),glass,air))),CartesianGrid.from_spacing(1.,4))
    with pytest.raises(ConvergenceError):
        propagate_interfaces(plane_wave(),assembly,CartesianGrid.from_spacing(1.,4),max_iterations=1,method='successive')
