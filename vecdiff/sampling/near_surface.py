"""Target-centred quadrature on a circular aperture of a surface of revolution.

Polar rays start at the projected target. Geometric radial panels resolve the
near singular peak; uniform panels independently resolve oscillatory far-field
contributions. This is ordinary off-surface quadrature, not a boundary-limit
formula or a dielectric boundary solver.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from .surface_sampling import SurfaceSampling
from ..surfaces.axisymmetric import AxisymmetricSurface


def sample_near_surface(surface, aperture, center, distance, *, radial_panels,
                        order=8, nphi=128, geometric_panels=18):
    """Disk quadrature centred at local (x,y), with clustering scale distance.

All lengths use the surface's unit. The target projection must lie strictly
inside the disk. radial_panels and nphi must also resolve the wavelength;
distance alone does not determine an adequate oscillatory quadrature.
    """
    if not isinstance(surface,AxisymmetricSurface):
        raise TypeError('requires an axisymmetric graph surface')
    center=np.asarray(center,float)
    if center.shape!=(2,) or not np.isfinite(center).all():
        raise ValueError('center must be a finite local two-vector')
    if not np.isfinite([aperture,distance]).all() or min(aperture,distance)<=0 or np.linalg.norm(center)>=aperture:
        raise ValueError('positive aperture/distance and a center strictly inside the aperture required')
    for name,value,minimum in [('radial_panels',radial_panels,1),('order',order,2),('nphi',nphi,4),('geometric_panels',geometric_panels,2)]:
        if not isinstance(value,(int,np.integer)) or value<minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    phi=np.arange(nphi)*2*np.pi/nphi; ex,ey=np.cos(phi),np.sin(phi)
    cx,cy=center; projection=cx*ex+cy*ey
    limit=-projection+np.sqrt(projection**2+aperture**2-cx*cx-cy*cy)
    fractions=np.unique(np.r_[np.linspace(0,1,radial_panels+1),
            np.geomspace(min(distance/aperture/4,.01),1,geometric_panels)])
    t,w=leggauss(order); a,b=fractions[:-1],fractions[1:]
    radial=((t[:,None]+1)*(b-a)/2+a).T.reshape(-1)
    wr=(w[:,None]*(b-a)/2).T.reshape(-1)
    r=radial[:,None]*limit; weights=wr[:,None]*limit
    x,y=cx+r*ex,cy+r*ey; rho=np.hypot(x,y)
    z=surface.sag(rho); slope=surface.slope(rho)
    sx=np.divide(slope*x,rho,out=np.zeros_like(x),where=rho!=0)
    sy=np.divide(slope*y,rho,out=np.zeros_like(y),where=rho!=0)
    jac=np.sqrt(1+sx*sx+sy*sy)
    normal=np.stack((-sx,-sy,np.ones_like(x)),axis=-1)/jac[...,None]
    p=np.stack((x,y,z),axis=-1)
    return SurfaceSampling(surface,surface.frame.points(p).reshape(-1,3),
        surface.frame.vectors(normal).reshape(-1,3),(weights*r*jac*(2*np.pi/nphi)).ravel())
