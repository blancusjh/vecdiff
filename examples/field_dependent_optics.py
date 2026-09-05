"""Macroscopic refraction/reflection for distinct distant object directions.

Every source direction is transformed independently on the physical surface.
This is an angle-dependent response, not convolution with a shifted on-axis PSF.
The two examples are single dielectric surfaces, not complete instruments.
"""
import numpy as np
from vecdiff import EvenAsphere,Medium,DielectricInterface,plane_wave,sample_surface,interface_transform

WAVELENGTH=.000193368  # mm


def configuration(kind):
    if kind=='refraction':
        return EvenAsphere(-.1,-2.25),Medium(1.5),Medium(),12.,np.array([0.,0.,20.]),30.
    if kind=='reflection':
        return EvenAsphere(-.1,-1.),Medium(),Medium(1.5),4.,np.array([0.,0.,-5.]),5.
    raise ValueError('kind must be refraction or reflection')


def response(kind, angle_degrees, *, nr=128, nphi=256):
    surface,n1,n2,aperture,focus,mapping=configuration(kind)
    angle=np.deg2rad(angle_degrees)
    source=plane_wave((np.sin(angle),0,np.cos(angle)),(np.cos(angle),0,-np.sin(angle)),
                      wavelength=WAVELENGTH,medium=n1)
    sampling=sample_surface(surface,(0,aperture),(0,2*np.pi),nr,nphi)
    out=interface_transform(source,DielectricInterface(surface,n1,n2),sampling)
    radiation=out.transmitted if kind=='refraction' else out.reflected
    center=focus+np.array([mapping*np.tan(angle),0.,0.])
    return radiation,center
