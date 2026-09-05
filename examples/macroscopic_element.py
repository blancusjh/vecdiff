"""A two-curved-interface stigmatic element and its off-axis response (mm)."""
import numpy as np
from vecdiff import EvenAsphere,SphericalCap,Frame,Medium,DielectricInterface,InterfaceAssembly,plane_wave,sample_surface
from vecdiff.propagation.high_frequency import propagate_high_frequency
from vecdiff.fields.eikonal_field import EikonalElectricField


def configuration():
    first=EvenAsphere(.1,-(1/1.5)**2)
    second=SphericalCap(12.,Frame(origin=[0,0,10.]))
    assembly=InterfaceAssembly((DielectricInterface(first,Medium(),Medium(1.5)),
                               DielectricInterface(second,Medium(1.5),Medium())))
    return assembly,4.,np.array([0.,0.,40.])


def response(angle_degrees=0.,*,wavelength=.000193368,nr=64,nphi=128,derivative_step=None,waist=None):
    assembly,aperture,focus=configuration(); angle=np.deg2rad(angle_degrees)
    incident=plane_wave((np.sin(angle),0,np.cos(angle)),(np.cos(angle),0,-np.sin(angle)),wavelength=wavelength)
    if waist is not None:
        if not np.isfinite(waist) or waist<=0:raise ValueError('waist must be finite and positive')
        direction=incident.wavevectors[0].real* wavelength/(2*np.pi)
        polarization=incident.amplitudes[0]
        incident=EikonalElectricField(lambda p:p@direction,lambda p:direction,
            lambda p:np.exp(-(np.sum(p*p,axis=-1)-(p@direction)**2)/waist**2)[...,None]*polarization,
            wavelength)
    sampling=sample_surface(assembly.interfaces[0].surface,(0,aperture),(0,2*np.pi),nr,nphi)
    return propagate_high_frequency(incident,assembly,sampling,derivative_step=derivative_step,apertures=(4.,3.5)),focus


def finite_conjugate_response(source_point=(0.,0.,-20.),*,wavelength=.000193368,nr=64,nphi=128):
    """Recover finite-conjugate stigmatism; displaced sources use the same faces."""
    from vecdiff import CartesianOval
    source_point=np.asarray(source_point,float)
    if source_point.shape!=(3,) or not np.isfinite(source_point).all() or source_point[2]>=0:
        raise ValueError('source must be a finite point before the first vertex')
    first=CartesianOval(-20.,30.,1.,1.5)
    second=SphericalCap(12.,Frame(origin=[0,0,10.]))
    assembly=InterfaceAssembly((DielectricInterface(first,Medium(),Medium(1.5)),
                               DielectricInterface(second,Medium(1.5),Medium())))
    def direction(p):
        r=p-source_point;return r/np.linalg.norm(r,axis=-1)[...,None]
    def envelope(p):
        u=direction(p);return 20*(np.array([1.,0.,0.])-u*u[...,0,None])/np.linalg.norm(p-source_point,axis=-1)[...,None]
    source=EikonalElectricField(lambda p:np.linalg.norm(p-source_point,axis=-1),direction,envelope,wavelength)
    q=sample_surface(first,(0,1.5),(0,2*np.pi),nr,nphi)
    return propagate_high_frequency(source,assembly,q,apertures=(1.5,1.5)),np.array([0.,0.,40.])
