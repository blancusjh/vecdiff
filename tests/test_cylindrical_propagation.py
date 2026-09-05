import numpy as np
import pytest
from vecdiff import Medium, Frame, SphericalCap, DielectricInterface, plane_wave, sample_surface, interface_transform


@pytest.mark.parametrize('direction',[-1,1])
def test_analytic_azimuth_matches_resolved_plane_wave_sum(direction):
    cap=SphericalCap(5.,Frame(origin=[.3,-.4,.7]))
    sampling=sample_surface(cap,(0,2),(0,2*np.pi),32,48)
    result=interface_transform(plane_wave(polarization=(1,.3j,0)),DielectricInterface(cap,Medium(),Medium(1.5)),sampling)
    rad=result.transmitted if direction==1 else result.reflected
    points=np.array([[.8,.2,3*direction],[2.,-1.,4*direction],[-1.,.6,2*direction]])
    analytic=rad.evaluate_propagating(points,direction=direction,n_theta=100,radial_count=1200,max_order=2)
    explicit=rad.angular_spectrum(direction=direction,n_theta=100,n_phi=128,backend='polar',radial_count=1200,max_order=2).evaluate(points)
    for a,b in zip(analytic,explicit):
        np.testing.assert_allclose(a,b,rtol=2e-11,atol=2e-12)


def test_unresolved_current_harmonics_are_rejected():
    cap=SphericalCap(5.)
    sampling=sample_surface(cap,(0,2),(0,2*np.pi),24,48)
    result=interface_transform(plane_wave(),DielectricInterface(cap,Medium(),Medium(1.5)),sampling)
    with pytest.raises(ValueError,match='azimuthal truncation'):
        result.transmitted.evaluate_propagating([[1,0,3]],max_order=0)
