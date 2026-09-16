"""Explicit periodic, isoplanatic image experiment helpers.

These helpers apply a supplied vector transfer function; they do not construct
an optical-system prescription or imply its validity away from the chosen field.
Source weights are normalized. Mutually incoherent source fields are never added.
"""
import numpy as np


def circuit_pattern(x):
    """Original geometric amplitude pattern at image scale (coordinates in µm)."""
    X,Y=np.meshgrid(x,x)
    mask=np.zeros(X.shape)
    rectangles=[(-1.8,-1.55,-1.8,1.8),(-1.8,.2,1.55,1.8),(-1.8,.2,-1.8,-1.55),
                (-.6,-.35,-1.,1.),(-.6,1.8,.75,1.),(.35,.6,-1.8,.15),(.35,1.8,-1.8,-1.55)]
    for x0,x1,y0,y1 in rectangles: mask[(X>=x0)&(X<x1)&(Y>=y0)&(Y<y1)]=1
    for cx in [.95,1.55]:
        for cy in [-.8,-.2]: mask[(abs(X-cx)<.15)&(abs(Y-cy)<.15)]=1
    return mask


def coherent_image(mask, transfer, source=(0,0)):
    """Periodic vector convolution for one exact integer-frequency source.

mask[y,x], transfer[y-frequency,x-frequency,component], unshifted FFT order.
source=(kx_bin,ky_bin); multiply mask by exp(+2pi i source.r/period).
The returned phasor uses the ordinary array origin; its constant source phase
has no effect on the incoherent sum. No normalization of output peaks is done.
    """
    mask=np.asarray(mask); transfer=np.asarray(transfer)
    if mask.ndim!=2 or transfer.shape[:2]!=mask.shape or transfer.ndim!=3:
        raise ValueError('mask=(ny,nx), transfer=(ny,nx,components) required')
    if not np.isfinite(mask).all() or not np.isfinite(transfer).all():
        raise ValueError('finite mask and transfer required')
    if len(source)!=2 or any(int(s)!=s for s in source):
        raise ValueError('source coordinates must be integer Fourier bins')
    shifted=np.roll(np.fft.fft2(mask), (int(source[1]),int(source[0])),axis=(0,1))
    return np.fft.ifft2(shifted[...,None]*transfer,axes=(0,1))


def aerial_image(mask, transfer, sources=((0,0),), weights=None):
    sources=np.asarray(sources)
    weights=np.ones(len(sources)) if weights is None else np.asarray(weights,float)
    if weights.shape!=(len(sources),) or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:
        raise ValueError('source weights must be finite, nonnegative and have positive sum')
    result=np.zeros(transfer.shape,float)
    for source,weight in zip(sources,weights/weights.sum()):
        result+=weight*abs(coherent_image(mask,transfer,source))**2
    return result


def disk_sources(radius_bins, step=1):
    """Equal-area Cartesian quadrature for a disk source, in Fourier bins."""
    b=int(np.floor(radius_bins/step))
    axis=np.arange(-b,b+1)*step
    return np.array([(x,y) for y in axis for x in axis if x*x+y*y<=radius_bins**2],int)
