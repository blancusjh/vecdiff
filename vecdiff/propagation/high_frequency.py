"""Leading high-frequency transport of separate spectral phase components.

A supplied EikonalElectricField uses one ray at each initial surface sample.
An ElectricSpectrum transports each incident plane-wave mode separately.
Vector Fresnel laws act before modes are summed. Homogeneous transport carries
optical path and ray-tube spreading; the final field is the existing spectral/
Green radiation of the resulting currents. This is an explicit eikonal
approximation between interfaces, NOT an exact Maxwell boundary solution.
It omits upstream edge diffraction and rejects intermediate caustics, TIR,
vignetting and non-graph surfaces. No translated point response is used.
"""
from dataclasses import dataclass
import numpy as np
from ..interfaces.fresnel import solve
from ..fields.eikonal_field import EikonalElectricField
from ..surfaces.axisymmetric import AxisymmetricSurface
from ..surfaces.surface import Plane
from ..sampling.surface_sampling import SurfaceSampling
from .interface_transform import BoundaryData
from .surface_radiation import SurfaceRadiation


@dataclass(frozen=True)
class ModalEncounter:
    sampling: object
    boundary: object
    optical_path: np.ndarray
    incident_direction: np.ndarray
    transmitted_direction: np.ndarray


@dataclass(frozen=True)
class HighFrequencyResult:
    transmitted: object
    reflected: tuple
    modes: tuple
    derivative_step: float


def _graph(surface, xy):
    if not isinstance(surface,(AxisymmetricSurface,Plane)):
        raise TypeError('high-frequency transport requires placed graph surfaces')
    xy=np.asarray(xy,float); r=np.linalg.norm(xy,axis=-1)
    if isinstance(surface,Plane):z=np.zeros_like(r);slope=np.zeros_like(r)
    else:z=surface.sag(r);slope=surface.slope(r)
    grad=np.divide(slope[...,None]*xy,r[...,None],out=np.zeros_like(xy),where=r[...,None]!=0)
    n=np.concatenate((-grad,np.ones(r.shape+(1,))),axis=-1)
    n/=np.linalg.norm(n,axis=-1)[...,None]
    return surface.frame.points(np.concatenate((xy,z[...,None]),axis=-1)),surface.frame.vectors(n)


def _intersect(p,u,surface,scale):
    frame=surface.frame; a=(p-frame.origin)@frame.rotation; b=u@frame.rotation
    if np.any(abs(b[...,2])<1e-10):raise ValueError('ray is parallel to the destination vertex plane')
    length=-a[...,2]/b[...,2]
    for _ in range(40):
        q=a+length[...,None]*b; r=np.linalg.norm(q[...,:2],axis=-1)
        sag=np.zeros_like(r) if isinstance(surface,Plane) else surface.sag(r)
        slope=np.zeros_like(r) if isinstance(surface,Plane) else surface.slope(r)
        dr=np.divide(np.sum(q[...,:2]*b[...,:2],axis=-1),r,out=np.zeros_like(r),where=r!=0)
        derivative=b[...,2]-slope*dr
        if np.any(abs(derivative)<1e-10):raise ValueError('grazing destination intersection')
        step=(q[...,2]-sag)/derivative; length-=step
        if np.max(abs(step))<2e-13*scale:break
    else:raise ValueError('destination intersection did not converge')
    if np.any(length<=1e-12*scale):raise ValueError('destination must be ahead of every transported ray')
    return p+length[...,None]*u,length


def _derivatives(a,h):return (a[1]-a[2])/(2*h),(a[3]-a[4])/(2*h)


def _reject_caustic(p,u,length,h):
    pu,pv=_derivatives(p,h); uu,uv=_derivatives(u,h); v=u[0]
    c=np.sum(v*np.cross(pu,pv),axis=-1)
    b=np.sum(v*(np.cross(uu,pv)+np.cross(pu,uv)),axis=-1)
    a=np.sum(v*np.cross(uu,uv),axis=-1)
    # Signed projected ray-tube area is c+b*l+a*l² along the segment.
    if np.any(abs(c)<1e-12):raise ValueError('singular incident ray tube')
    b=b/c; a=a/c; distance=length[0]
    endpoint=1+b*distance+a*distance**2
    vertex=np.divide(-b,2*a,out=np.zeros_like(b),where=a!=0)
    minimum=np.minimum(1.,endpoint)
    inside=(a>0)&(vertex>0)&(vertex<distance)
    minimum=np.where(inside,np.minimum(minimum,1+b*vertex+a*vertex**2),minimum)
    if np.any(minimum<=1e-8):raise ValueError('intermediate caustic: high-frequency ray-tube transport is not valid')


def _geometry(carrier,interfaces,xy,h,scale):
    if not all(isinstance(i.surface,(AxisymmetricSurface,Plane)) for i in interfaces):
        raise TypeError("high-frequency transport requires placed graph surfaces")
    offsets=np.array([[0,0],[h,0],[-h,0],[0,h],[0,-h]])
    p,n=_graph(interfaces[0].surface,np.asarray(xy)[None]+offsets[:,None])
    u=carrier.directions(p).copy(); geometry=[]
    for j,interface in enumerate(interfaces):
        local=(p-interface.surface.frame.origin)@interface.surface.frame.rotation
        _,normal=_graph(interface.surface,local[...,:2]); normal*=interface.normal_sign
        # Fresnel/Snell physics remains in interfaces/fresnel.py.
        f=solve(u*interface.incident_medium.n,np.zeros_like(u),normal,
                interface.incident_medium,interface.transmitted_medium,wavelength=2*np.pi)
        if np.any(abs(f.transmitted_k.imag)>1e-12):raise ValueError('TIR/evanescent transmission is outside high-frequency ray transport')
        transmitted=f.transmitted_k.real/interface.transmitted_medium.n
        if np.any(np.sum(transmitted*normal,axis=-1)<1e-8):raise ValueError('critical transmission is outside high-frequency transport')
        pu,pv=_derivatives(p,h); area=np.cross(pu,pv)
        geometry.append((p,u,transmitted,normal,area))
        if j+1<len(interfaces):
            q,length=_intersect(p,transmitted,interfaces[j+1].surface,scale)
            _reject_caustic(p,transmitted,length,h)
            p,u=q,transmitted
    return geometry


def _mode(carrier,interfaces,xy,weights,h,scale,apertures=None):
    wavelength=carrier.wavelength
    geometry=_geometry(carrier,interfaces,xy,h,scale); records=[]
    first_area=np.linalg.norm(geometry[0][4],axis=-1)
    envelope=carrier.amplitudes(geometry[0][0][0])
    optical_path=np.asarray(carrier.optical_path(geometry[0][0][0]),float)
    if optical_path.shape!=envelope.shape[:-1] or not np.isfinite(optical_path).all():raise ValueError("optical_path must return finite scalar lengths")
    for j,(interface,g) in enumerate(zip(interfaces,geometry)):
        p,u,t,n,area=g; jac=np.linalg.norm(area,axis=-1)
        if apertures is not None and apertures[j] is not None:
            local=(p[0]-interface.surface.frame.origin)@interface.surface.frame.rotation
            if np.any(np.linalg.norm(local[:,:2],axis=-1)>apertures[j]):
                raise ValueError('vignetting requires explicit aperture diffraction; reduce the input domain')
        if j:
            previous=geometry[j-1]
            before=np.linalg.norm(previous[4],axis=-1)*np.abs(np.sum(previous[2][0]*previous[3][0],axis=-1))
            after=jac*np.abs(np.sum(u[0]*n[0],axis=-1))
            envelope*=np.sqrt(before/after)[:,None]
            optical_path=optical_path+interface.incident_medium.n*np.linalg.norm(p[0]-previous[0][0],axis=-1)
        f=solve(u[0]*interface.incident_medium.wavenumber(wavelength),envelope,n[0],
                interface.incident_medium,interface.transmitted_medium,wavelength)
        phase=np.exp(2j*np.pi*optical_path/wavelength)[:,None]
        ei=envelope*phase; er=f.reflected_E*phase; et=f.transmitted_E*phase
        hi=interface.incident_medium.n*np.cross(u[0],ei)
        hr=np.cross(f.reflected_k,er)/(2*np.pi/wavelength)
        ht=np.cross(f.transmitted_k,et)/(2*np.pi/wavelength)
        sampling=SurfaceSampling(interface.surface,p[0],n[0]*interface.normal_sign,weights*jac/first_area)
        records.append(ModalEncounter(sampling,BoundaryData(ei,hi,er,hr,et,ht),optical_path.copy(),u[0],t[0]))
        envelope=f.transmitted_E
    return tuple(records)


def _radiation(records,interface,wavelength,reflected):
    samples=[r.sampling for r in records]
    q=SurfaceSampling(interface.surface,np.concatenate([s.points for s in samples]),
                      np.concatenate([s.normals for s in samples]),np.concatenate([s.weights for s in samples]))
    e=np.concatenate([r.boundary.reflected_E if reflected else r.boundary.transmitted_E for r in records])
    h=np.concatenate([r.boundary.reflected_H if reflected else r.boundary.transmitted_H for r in records])
    return SurfaceRadiation.from_boundary(q,e,h,wavelength,
            interface.incident_medium if reflected else interface.transmitted_medium,
            normal_sign=(-1 if reflected else 1)*interface.normal_sign)


def _carriers(incident):
    if isinstance(incident,EikonalElectricField):return (incident,)
    if np.any(incident.wavevectors.imag):raise ValueError('high-frequency transport requires propagating incident modes')
    result=[]; k0=2*np.pi/incident.wavelength
    for k,a in zip(incident.wavevectors.real,incident.amplitudes):
        if np.any(a):
            result.append(EikonalElectricField(lambda p,k=k: p@k/k0,
                lambda p,k=k: k/k0,lambda p,a=a:a,incident.wavelength,incident.medium))
    return tuple(result)


def propagate_high_frequency(incident,assembly,sampling,*,derivative_step=None,apertures=None):
    """Transport distinct incident modes through a forward dielectric path.

One supplied phase component uses one ray per sample, plus four differential rays used only to
measure geometric spreading. This is O(modes * samples * interfaces), with
no wavelength-scale aperture raster. The final SurfaceRadiation keeps vector
diffraction. Halve derivative_step and refine the initial surface quadrature
independently; neither check bounds the omitted wave correction between faces.

Intermediate caustics, evanescent incidence/transmission and vignetting reject.
Reflections are returned at each encounter, but are not fed back into the path.
    """
    interfaces=assembly.interfaces
    if sampling.surface is not interfaces[0].surface:raise ValueError('initial sampling belongs to another interface')
    if incident.medium!=interfaces[0].incident_medium:raise ValueError('incident medium does not match assembly')
    local=(sampling.points-sampling.surface.frame.origin)@sampling.surface.frame.rotation
    scale=max(float(np.ptp(local[:,:2],axis=0).max()),
              max(np.linalg.norm(i.surface.frame.origin-sampling.surface.frame.origin) for i in interfaces),1e-30)
    h=1e-5*scale if derivative_step is None else float(derivative_step)
    if not np.isfinite(h) or h<=0:raise ValueError('derivative_step must be finite and positive')
    if apertures is not None:
        apertures=tuple(apertures)
        if len(apertures)!=len(interfaces) or any(a is not None and (not np.isfinite(a) or a<=0) for a in apertures):
            raise ValueError('one positive aperture or None is required per interface')
    modes=[]
    for carrier in _carriers(incident):
        modes.append(_mode(carrier,interfaces,local[:,:2],sampling.weights,h,scale,apertures))
    if not modes:raise ValueError('at least one nonzero incident mode is required')
    transmitted=_radiation([m[-1] for m in modes],interfaces[-1],incident.wavelength,False)
    reflected=tuple(_radiation([m[j] for m in modes],i,incident.wavelength,True) for j,i in enumerate(interfaces))
    return HighFrequencyResult(transmitted,reflected,tuple(modes),h)


def sample_high_frequency(incident,assembly,xy,*,derivative_step,initial_aperture=None):
    """Reconstruct incident E,H at prescribed points of the last surface.

Invert the ray map independently for each incident phase component. This is
useful for comparison with the full radiation from the preceding interface.
The output is BEFORE the last Fresnel encounter. No interpolated total-field
ray or phase unwrapping is used. Intermediate caustics reject as above.
    """
    interfaces=assembly.interfaces; xy=np.asarray(xy,float)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)==0 or not np.isfinite(xy).all():raise ValueError('xy must be finite (samples,2)')
    if initial_aperture is not None and (not np.isfinite(initial_aperture) or initial_aperture<=0):
        raise ValueError("initial_aperture must be finite and positive")
    h=float(derivative_step)
    if not np.isfinite(h) or h<=0:raise ValueError('derivative_step must be finite and positive')
    if incident.medium!=interfaces[0].incident_medium:raise ValueError('incompatible incident medium')
    scale=max(np.linalg.norm(i.surface.frame.origin-interfaces[0].surface.frame.origin) for i in interfaces)
    scale=max(scale,float(np.max(abs(xy))),h)
    e=np.zeros((len(xy),3),complex); magnetic=e.copy()
    for carrier in _carriers(incident):
        initial=xy.copy()
        for _ in range(20):
            g=_geometry(carrier,interfaces,initial,h,scale)
            destination=(g[-1][0]-interfaces[-1].surface.frame.origin)@interfaces[-1].surface.frame.rotation
            du,dv=_derivatives(destination,h)
            matrix=np.stack((du[:,:2],dv[:,:2]),axis=-1)
            step=np.linalg.solve(matrix,(destination[0,:,:2]-xy)[...,None])[...,0]
            initial-=step
            if np.max(abs(step))<1e-11*scale:break
        else:raise ValueError('inverse ray map did not converge')
        records=_mode(carrier,interfaces,initial,np.ones(len(xy)),h,scale)
        keep=np.ones(len(xy),bool) if initial_aperture is None else np.linalg.norm(initial,axis=-1)<=initial_aperture
        e[keep]+=records[-1].boundary.incident_E[keep]; magnetic[keep]+=records[-1].boundary.incident_H[keep]
    return e,magnetic
