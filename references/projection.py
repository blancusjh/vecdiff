"""Independent sine-condition pupil reference for local DUV image formation.

Uniform entrance-pupil illumination; Cartesian image angular-spectrum density
contains 1/sqrt(cos theta): sqrt(cos theta) Debye apodization divided by the
cos(theta) Jacobian. This reference is not a per-interface propagation routine.
"""
import numpy as np


def pupil_transfer(fx,fy,*,wavelength,na,index,polarization=(1,0),wavefront=None,defocus=0.):
    if not 0<na<index or wavelength<=0:
        raise ValueError('requires positive wavelength and 0 < NA < image index')
    fx,fy=np.broadcast_arrays(fx,fy)
    rho=np.hypot(fx,fy)*wavelength/na
    inside=rho<=1
    phi=np.arctan2(fy,fx)
    s=np.where(inside,rho*na/index,0.); c=np.sqrt(1-s*s)
    cp,sp=np.cos(phi),np.sin(phi)
    ex,ey=np.asarray(polarization,complex)
    meridional=ex*cp+ey*sp; azimuthal=-ex*sp+ey*cp
    e=np.stack((meridional*c*cp-azimuthal*sp,meridional*c*sp+azimuthal*cp,-meridional*s),axis=-1)
    W=np.zeros_like(rho) if wavefront is None else wavefront(np.where(inside,rho*cp,0),np.where(inside,rho*sp,0))
    phase=np.exp(2j*np.pi*W+2j*np.pi*index/wavelength*c*defocus)
    e*= (inside*phase/np.sqrt(c))[...,None]
    k=2*np.pi*index/wavelength*np.stack((s*cp,s*sp,c),axis=-1)
    return e,k,inside
