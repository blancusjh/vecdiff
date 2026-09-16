"""A 10 mm stigmatic conic at DUV wavelength, using per-k Fresnel traces.

The local spectrum is a bounded expansion of SurfaceRadiation about the focus.
The independent comparison here is the full dyadic radiation of the same traces;
this does not close the curved Maxwell boundary-validation problem.
"""
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Ellipse
from vecdiff import EvenAsphere, Medium, DielectricInterface, plane_wave, sample_surface, interface_transform
from vecdiff.fields.polarization import stokes


WAVELENGTH = 193.368e-6  # mm
RADIUS = 10.0  # mm, signed curvature radius is -RADIUS


def radiation(radius=RADIUS, wavelength=WAVELENGTH, nr=32, nphi=64,
              polarization=(1, 0, 0)):
    surface = EvenAsphere(-1/radius, -2.25)
    samples = sample_surface(surface, (0, 1.2*radius), (0, 2*np.pi), nr, nphi)
    incoming = plane_wave(wavelength=wavelength, medium=Medium(1.5), polarization=polarization)
    return interface_transform(incoming, DielectricInterface(surface, Medium(1.5), Medium()), samples).transmitted


def relative_eh(pair, reference):
    return float(np.linalg.norm(np.concatenate([a-b for a,b in zip(pair,reference)],axis=-1))/
                 np.linalg.norm(np.concatenate(reference,axis=-1)))


def run():
    start = perf_counter()
    rad = radiation()
    center = np.array([0., 0., 2*RADIUS])
    local = rad.local_spectrum(center, 13*WAVELENGTH)
    construction = perf_counter()-start
    x = np.linspace(-3,3,241); z = np.linspace(-12,12,301)
    X,Y = np.meshgrid(x,x); XM,Z = np.meshgrid(x,z)
    transverse = center+WAVELENGTH*np.stack((X,Y,0*X),axis=-1)
    meridional = center+WAVELENGTH*np.stack((XM,0*XM,Z),axis=-1)
    start=perf_counter()
    e,h=local.evaluate(transverse,backend='nufft')
    em,hm=local.evaluate(meridional,backend='nufft')
    evaluation=perf_counter()-start
    intensity=np.sum(abs(e)**2,axis=-1); peak=float(intensity.max())
    im=np.sum(abs(em)**2,axis=-1)
    # Held-out points include off-axis and defocused observations, with no
    # fitted phase or fitted normalization in the complex-field comparison.
    offsets=np.array([[a,b,c] for a in [-3,-1.5,0,1.5,3] for b in [-1,1] for c in [-12,-6,0,6,12]])
    points=center+WAVELENGTH*offsets
    start=perf_counter(); reference=rad.evaluate(points); direct_time=perf_counter()-start
    prediction=local.evaluate(points)
    refined=radiation(nr=48,nphi=96).evaluate(points)
    error=relative_eh(prediction, reference)
    quadrature=relative_eh(reference,refined)
    fourier=relative_eh(local.evaluate(points,backend='nufft'),prediction)
    # Circular input makes the spatial polarization ellipse structure visible.
    circular=radiation(polarization=np.array([1,1j,0])/np.sqrt(2)).local_spectrum(center,13*WAVELENGTH)
    ec,_=circular.evaluate(transverse,backend='nufft')
    S=stokes(ec[...,0],ec[...,1]); chi=.5*np.arcsin(np.clip(S[...,3]/np.maximum(S[...,0],1e-300),-1,1))
    psi=.5*np.arctan2(S[...,2],S[...,1]); ic=np.sum(abs(ec)**2,axis=-1)
    fig,axes=plt.subplots(2,3,figsize=(13.5,8),constrained_layout=True)
    panels=[(axes[0,0],intensity/peak,[-3,3,-3,3],'Transverse total',r'$y/\lambda$'),
            (axes[0,1],abs(e[...,2])**2/peak,[-3,3,-3,3],'Transverse longitudinal',r'$y/\lambda$'),
            (axes[1,0],im/peak,[-3,3,-12,12],'Meridional total',r'$(z-f)/\lambda$'),
            (axes[1,1],abs(em[...,2])**2/peak,[-3,3,-12,12],'Meridional longitudinal',r'$(z-f)/\lambda$')]
    for ax,values,extent,title,ylabel in panels:
        picture=ax.imshow(values,origin='lower',extent=extent,cmap='hot',norm=PowerNorm(.5,0,.12 if 'longitudinal' in title else 1),aspect='auto')
        ax.set(xlabel=r'$x/\lambda$',ylabel=ylabel,title=title+' | linear input')
        fig.colorbar(picture,ax=ax,label=r'$|E|^2$ or $|E_z|^2$ / focal peak')
    ax=axes[0,2]
    image=ax.imshow(np.where(ic>.005*ic.max(),np.degrees(chi),np.nan),extent=[-3,3,-3,3],origin='lower',cmap='coolwarm',vmin=-45,vmax=45)
    for iy in range(6,len(x),14):
        for ix in range(6,len(x),14):
            if ic[iy,ix] < .005*ic.max():continue
            width=.16
            ax.add_patch(Ellipse((x[ix],x[iy]),width,width*abs(np.tan(chi[iy,ix])),
                                angle=np.degrees(psi[iy,ix]),facecolor='none',edgecolor='black',linewidth=.6))
    ax.set(xlabel=r'$x/\lambda$',ylabel=r'$y/\lambda$',title='Transverse ellipses | circular input')
    fig.colorbar(image,ax=ax,label='Ellipticity angle (degrees)')
    ax=axes[1,2]
    cut=np.linspace(-3,3,101); p=center+WAVELENGTH*np.c_[cut,cut*0,cut*0]
    exact=rad.evaluate(p)[0]; approximate=local.evaluate(p)[0]
    ax.plot(cut,np.sum(abs(exact)**2,axis=-1)/peak,color='black',label='Full radiation kernel')
    ax.plot(cut[::4],np.sum(abs(approximate[::4])**2,axis=-1)/peak,'o',mfc='none',label='Local spectral expansion')
    ax.set(xlabel=r'$x/\lambda$',ylabel='Total electric norm / same focal peak',title='Same per-k Fresnel boundary traces')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Stigmatic conic: |R|=10 mm, aperture radius=12 mm, f=20 mm, λ=193.368 nm\n'
                 'n=1.5 → 1; no ideal-pupil or Richards–Wolf amplitudes',fontsize=13)
    report=dict(observation_ball_radius_wavelengths=13,wavelength_mm=WAVELENGTH,radius_mm=RADIUS,
                radius_over_wavelength=RADIUS/WAVELENGTH,aperture_radius_mm=12.,focus_mm=20.,
                source_nodes=32*64,transverse_shape=list(X.shape),meridional_shape=list(XM.shape),
                construction_seconds=construction,field_maps_seconds=evaluation,reference_points_seconds=direct_time,
                kernel_relative_EH_error=error,surface_quadrature_relative_EH_change=quadrature,
                nufft_relative_EH_error=fourier,electric_absolute_error_bound=local.electric_error_bound,
                magnetic_absolute_error_bound=local.magnetic_error_bound,
                electric_error_bound_over_peak_amplitude=float(local.electric_error_bound/np.sqrt(peak)),
                longitudinal_norm_fraction=float(np.sum(abs(e[...,2])**2)/np.sum(intensity)),
                parameters=dict(indices=[1.5,1.],conic=-2.25,source_quadrature=[32,64],length_unit='mm'),
                assumptions=['Local expansion of the existing Fresnel-current radiation.',
                             'Kernel bound excludes source quadrature and dielectric trace error.',
                             'The geometrically stigmatic residual phase is smooth; no full DUV train is propagated.'],
                scope='Local radiation approximation; full dielectric boundary accuracy remains unvalidated.')
    return fig,report


if __name__=='__main__':
    import json
    from pathlib import Path
    fig,report=run();out=Path('examples/output/macroscopic_focus');out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'figure.png',dpi=180);(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
