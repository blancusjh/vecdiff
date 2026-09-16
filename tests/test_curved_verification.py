import numpy as np
import pytest
from vecdiff import EvenAsphere,Medium,DielectricInterface,plane_wave,sample_surface,interface_transform
from vecdiff.sampling.near_surface import sample_near_surface


def test_target_centered_disk_area_and_moments():
    surface=EvenAsphere()
    q=sample_near_surface(surface,2.,[.7,-.3],.001,radial_panels=12,order=8,nphi=128)
    np.testing.assert_allclose(q.weights.sum(),4*np.pi,rtol=2e-14)
    np.testing.assert_allclose(np.sum(q.points*q.weights[:,None],axis=0),0,atol=1e-13)
    np.testing.assert_allclose(np.sum(q.points[:,0]**2*q.weights),4*np.pi,rtol=2e-14)


def test_patch_evaluation_bounds_full_kernel_for_distinct_global_regions():
    s=EvenAsphere(-.01,-2.25)
    q=sample_surface(s,(0,80),(0,2*np.pi),24,48)
    rad=interface_transform(plane_wave(medium=Medium(1.5)),DielectricInterface(s,Medium(1.5),Medium()),q).transmitted
    p=np.array([[x,y,200+z] for x in [-8,-4,0,4,8] for y in [-1,1] for z in [-1,1]])
    result=rad.evaluate_local(p,radius=1.,backend='direct')
    exact=rad.evaluate(p)
    assert result.patch_count>1
    assert np.max(np.linalg.norm(result.electric-exact[0],axis=-1))<=result.electric_error_bound
    assert np.max(np.linalg.norm(result.magnetic-exact[1],axis=-1))<=result.magnetic_error_bound
    # The output retains absolute positions: opposite off-axis E_z has opposite sign.
    np.testing.assert_allclose(result.electric[0,2],-result.electric[-2,2],rtol=1e-10,atol=1e-10)


def test_invalid_near_quadrature_center():
    with pytest.raises(ValueError):
        sample_near_surface(EvenAsphere(),1.,[1.,0],.1,radial_panels=8)
