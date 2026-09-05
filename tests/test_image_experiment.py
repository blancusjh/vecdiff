"""Independent controls of source summation and image formation conventions."""
import numpy as np
from examples.image_formation import coherent_image,aerial_image


def test_unit_transfer_reproduces_mask_and_tilt():
    y,x=np.indices((24,32)); mask=np.exp(2j*np.pi*3*x/32)*(x<16)
    transfer=np.zeros((24,32,3),complex);transfer[...,0]=1
    field=coherent_image(mask,transfer,(2,-3))
    np.testing.assert_allclose(field[...,0],mask*np.exp(2j*np.pi*(2*x/32-3*y/24)),atol=1e-14)
    np.testing.assert_allclose(field[...,1:],0,atol=1e-14)


def test_incoherent_weights_and_single_source_limit():
    rng=np.random.default_rng(4); mask=rng.random((12,12))
    transfer=rng.random((12,12,3))+1j*rng.random((12,12,3))
    a=coherent_image(mask,transfer,(0,0));b=coherent_image(mask,transfer,(2,1))
    actual=aerial_image(mask,transfer,[(0,0),(2,1)],[1,3])
    np.testing.assert_allclose(actual,(abs(a)**2+3*abs(b)**2)/4,atol=1e-14)
    np.testing.assert_allclose(aerial_image(mask,transfer),abs(a)**2,atol=1e-14)


def test_single_spatial_frequency_measures_transfer():
    y,x=np.indices((16,16));mask=np.exp(2j*np.pi*(3*x+2*y)/16)
    transfer=np.zeros((16,16,3),complex);transfer[2,3]=[.3,.4j,.2]
    np.testing.assert_allclose(coherent_image(mask,transfer),mask[...,None]*transfer[2,3],atol=2e-15)


def test_duv_reference_is_transverse_and_preserves_pupil_flux_weight():
    from references.projection import pupil_transfer
    f=np.linspace(-8,8,47);fx,fy=np.meshgrid(f,f)
    e,k,inside=pupil_transfer(fx,fy,wavelength=.193368,na=1.2,index=1.59667693,polarization=np.array([1,1j])/np.sqrt(2))
    np.testing.assert_allclose(np.sum(k*e,axis=-1),0,atol=1e-13)
    cosine=k[...,2]/(2*np.pi*1.59667693/.193368)
    np.testing.assert_allclose(np.sum(abs(e)**2,axis=-1)*cosine,inside.astype(float),atol=2e-15)
