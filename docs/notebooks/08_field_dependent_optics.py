# %% [markdown]
# # Macroscopic optics with a response computed for each object direction
#
# **Use case:** compute off-axis vector fields and a small distant-source scene
# through centimetre-scale refracting and millimetre-scale reflecting surfaces at
# $\lambda_0=193.368$ nm. Every incident direction gets its own spectral Fresnel
# transformation on the actual surface. No convolution, invariant-shift assumption,
# or translated on-axis point response is used to form these results.
#
# This recovers and extends the earlier stigmatic single-surface use case to
# off-axis illumination. It is not complete multi-element DUV transport. The
# mirror is the dielectric reflection branch ($1\to1.5$), not a metal or coating.
# The reference theories remain outside the implementation.
# %%
from pathlib import Path
import sys,json,time
root=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'pyproject.toml').exists())
sys.path.insert(0,str(root))
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','inline')
import numpy as np
import matplotlib.pyplot as plt
from examples.notebook_tools import style,show,scalar_map,polarization_map
from examples.field_dependent_optics import configuration,response,WAVELENGTH
style()
# %% [markdown]
# ## 1. Actual refracting and reflecting geometries
# Both surfaces use
# $$z(r)=\frac{cr^2}{1+\sqrt{1-(1+K)c^2r^2}}.$$
# The exit diopter has $c=-0.1\,{\rm mm}^{-1}$, $K=-2.25$, aperture radius
# 12 mm, indices $1.5\to1$, and on-axis focus $z=20$ mm. The paraboloid has
# $K=-1$, aperture radius 4 mm, and reflected focus $z=-5$ mm.
# The drawn rays describe on-axis geometry; all field panels below come from
# the surface radiation integral. Aperture diameters exceed 124,000 and 41,000
# vacuum wavelengths respectively.
# %%
fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
for kind,ax in zip(['refraction','reflection'],axes):
    surface,n1,n2,aperture,focus,mapping=configuration(kind)
    rho=np.linspace(-aperture,aperture,601); sag=surface.sag(abs(rho))
    ax.plot(sag,rho,'k',lw=2,label='Physical surface')
    for r in np.linspace(-aperture,aperture,9):
        s=surface.sag(abs(r)); ax.plot([min(sag)-1,s],[r,r],color='tab:blue',alpha=.5)
        ax.plot([s,focus[2]],[r,0],color='tab:orange',alpha=.6)
    ax.plot(focus[2],0,'ro',label='On-axis focus')
    ax.set(xlabel='z (mm)',ylabel='x (mm)',title=f'{kind.capitalize()}: {n1.n} → {n2.n}',aspect='equal'); ax.legend(fontsize=8)
show(fig,'08_macroscopic_geometry')
# %% [markdown]
# ## 2. Compute each off-axis response independently
# For illumination angle $\theta$, the incident mode is
# $k=n_1k_0(\sin\theta,0,\cos\theta)$ and $E_0=(\cos\theta,0,-\sin\theta)$.
# We use a paraxial prediction only to choose the observation window. It does not
# translate any field. Every observation is evaluated at its global position.
#
# `evaluate_local` builds a fresh expansion of the fixed source currents for each
# occupied observation patch. Its absolute $E$ and $\mathcal H$ error bounds cover
# the Green-kernel expansion, not surface quadrature or dielectric trace accuracy.
# The patch radius is $4\lambda_0$; source quadrature must converge separately for
# every illumination and observation region.
# %%
angles=[0.,.002,.01,.02]
x=np.linspace(-15,15,301)*WAVELENGTH; y=np.linspace(-4,4,101)*WAVELENGTH
X,Y=np.meshgrid(x,y)
fields={}; lines={}; radiations={}; centers={}
start=time.perf_counter()
for kind in ['refraction','reflection']:
    for angle in angles:
        rad,center=response(kind,angle); radiations[kind,angle]=rad; centers[kind,angle]=center
        result=rad.evaluate_local(center+np.stack((X,Y,0*X),axis=-1),radius=4*WAVELENGTH)
        fields[kind,angle]=result.electric
        lines[kind,angle]=np.sum(abs(result.electric[len(y)//2])**2,axis=-1)
print(f'Computed {2*len(angles)*X.size:,} vector observations for eight distinct surface transformations in {time.perf_counter()-start:.2f} s')
fig,axes=plt.subplots(2,4,figsize=(17,7),layout='constrained')
for j,kind in enumerate(['refraction','reflection']):
    peak=np.sum(abs(fields[kind,0.])**2,axis=-1).max()
    for ax,angle in zip(axes[j],angles):
        scalar_map(fig,ax,np.sum(abs(fields[kind,angle])**2,axis=-1)/peak,x*1e3,y*1e3,
                   f'{kind}, θ={angle:g}°',xlabel='x − predicted image (µm)',ylabel='y (µm)',
                   label='Electric norm² / on-axis peak',vmax=1)
show(fig,'08_field_dependent_spots')
# %% [markdown]
# ## 3. Compare with a shifted on-axis template, without using it as a model
# The blue curve is a fresh calculation at the stated angle; the dashed curve is
# the on-axis result in the predicted shifted coordinate frame. No individual
# peak normalization hides throughput or peak changes. The mismatch includes
# both image displacement beyond the paraxial prediction and shape changes.
# %%
fig,axes=plt.subplots(2,3,figsize=(15,7),layout='constrained')
for row,kind in enumerate(['refraction','reflection']):
    peak=lines[kind,0.].max()
    for ax,angle in zip(axes[row],angles[1:]):
        ax.plot(x*1e3,lines[kind,angle]/peak,label='Recomputed surface response')
        ax.plot(x*1e3,lines[kind,0.]/peak,'--',alpha=.6,label='Shifted on-axis template')
        ax.set(xlabel='x − predicted image (µm)',ylabel='Electric norm² / on-axis peak',
               title=f'{kind}, θ={angle:g}°'); ax.legend(fontsize=8)
show(fig,'08_shift_comparison')
# %% [markdown]
# ## 4. Through-focus structure and polarization of the off-axis image
# The refracting conic at $0.002^\circ$ already departs strongly from its stigmatic
# on-axis image. Meridional coordinates are physical offsets from the predicted
# focus. Ellipses show the transverse polarization, with low-intensity pixels
# excluded; longitudinal electric energy is shown separately.
# %%
kind='refraction'; angle=.002; rad=radiations[kind,angle]; center=centers[kind,angle]
z=np.linspace(-10,10,151)*WAVELENGTH; xm=np.linspace(-6,9,151)*WAVELENGTH
XM,Z=np.meshgrid(xm,z)
meridional=rad.evaluate_local(center+np.stack((XM,0*XM,Z),axis=-1),radius=4*WAVELENGTH).electric
e=fields[kind,angle]; peak=np.sum(abs(fields[kind,0.])**2,axis=-1).max()
fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
scalar_map(fig,axes[0],np.sum(abs(meridional)**2,axis=-1)/peak,xm*1e3,z*1e3,'Off-axis meridional field',
           xlabel='x − predicted image (µm)',ylabel='z − f (µm)',label='Electric norm² / on-axis peak')
scalar_map(fig,axes[1],abs(e[...,2])**2/peak,x*1e3,y*1e3,'Longitudinal focal component',
           xlabel='x − predicted image (µm)',ylabel='y (µm)',label=r'$|E_z|^2$ / on-axis total peak')
polarization_map(fig,axes[2],e,x*1e3,y*1e3,title='Off-axis transverse polarization')
show(fig,'08_off_axis_vector_field')
# %% [markdown]
# ## 5. Form a scene by propagating each source, then combine fields
# Three equal-strength distant points at $-0.002^\circ,0,+0.002^\circ$ illuminate
# the exit diopter. They are evaluated on the **same global image coordinates**.
# Mutually incoherent sources give $I=\sum_j w_j|E_j|^2$; phase-locked sources give
# $I=|\sum_j\sqrt{w_j}e^{i\phi_j}E_j|^2$, here $w_j=1/3$, $\phi_j=0$.
# Equal illumination strength does not imply equal image-plane peak intensity.
# This small scene demonstrates field-dependent imaging directly; an extended
# lithographic mask through a full multi-element system remains a separate task.
# %%
scene_x=np.linspace(-3.5,3.5,301)*1e-3; scene_y=np.linspace(-.8,.8,101)*1e-3
SX,SY=np.meshgrid(scene_x,scene_y); scene_points=np.stack((SX,SY,20+0*SX),axis=-1)
scene_fields=[]
for angle in [-.002,0.,.002]:
    rad,center=response('refraction',angle)
    scene_fields.append(rad.evaluate_local(scene_points,radius=4*WAVELENGTH).electric)
scene_fields=np.asarray(scene_fields)
incoherent=np.mean(np.sum(abs(scene_fields)**2,axis=-1),axis=0)
coherent=np.sum(abs(np.sum(scene_fields,axis=0)/np.sqrt(3))**2,axis=-1)
normalization=max(incoherent.max(),coherent.max())
fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
for ax,values,title in zip(axes[:2],[incoherent,coherent],['Three mutually incoherent sources','Three phase-locked sources']):
    scalar_map(fig,ax,values/normalization,scene_x*1e3,scene_y*1e3,title,xlabel='Global image x (µm)',ylabel='y (µm)',label='Electric norm² / shared scene peak',vmax=1)
for field,angle in zip(scene_fields,[-.002,0.,.002]):
    axes[2].plot(scene_x*1e3,np.sum(abs(field[len(scene_y)//2])**2,axis=-1)/(3*normalization),label=f'{angle:g}°')
axes[2].set(xlabel='Global image x (µm)',ylabel='Electric norm² / shared scene peak',title='Individual incoherent contributions'); axes[2].legend()
show(fig,'08_direct_scene')
# %% [markdown]
# ## 6. Verify the numerical calculation at held-out global positions
# Direct dyadic-Green evaluation is independent of the local expansion. Doubling
# radial and azimuthal source resolution checks oscillatory surface quadrature.
# Neither comparison validates the local Fresnel boundary approximation itself;
# notebook 07 tests that separate physical requirement.
# %%
checks=[]
for kind,angle in [('refraction',.002),('reflection',.02)]:
    rad=radiations[kind,angle]; center=centers[kind,angle]
    held=center+WAVELENGTH*np.array([[-9,0,-3],[-2,1,2],[0,0,0],[3,-1,-2],[11,0,3]])
    fast=rad.evaluate_local(held,radius=4*WAVELENGTH)
    full=rad.evaluate(held)
    finer,_=response(kind,angle,nr=256,nphi=512); fine=finer.evaluate(held)
    relative=lambda a,b: np.linalg.norm(np.concatenate(a,axis=-1)-np.concatenate(b,axis=-1))/np.linalg.norm(np.concatenate(b,axis=-1))
    kernel=relative((fast.electric,fast.magnetic),full); quadrature=relative(full,fine)
    assert np.max(np.linalg.norm(fast.electric-full[0],axis=-1))<=fast.electric_error_bound
    assert np.max(np.linalg.norm(fast.magnetic-full[1],axis=-1))<=fast.magnetic_error_bound
    assert quadrature<1e-6
    checks.append((kind,kernel,quadrature))
    print(f'{kind}: relative E/H kernel error={kernel:.3e}; source-quadrature change={quadrature:.3e}; absolute bounds passed')
benchmark=json.loads((root/'benchmarks/results/field_dependent_optics.json').read_text())
for row in benchmark['cases']:
    assert row['absolute_kernel_bounds_passed'] and row['surface_quadrature_relative_EH_change']<1e-6
# %% [markdown]
# **What is established:** direct field-dependent macroscopic single-surface
# refraction and dielectric reflection, vector focal and meridional fields, and
# coherent/incoherent source combination without shift invariance. The reported
# speed applies to these smooth-phase near-focus patches and source samplings.
# Larger fields can require much finer source quadrature; a fast NUFFT alone does
# not resolve aliased surface phase. General extended surface-to-surface transport,
# complete instrument imaging, and uniformly verified curved-boundary accuracy
# remain pending. No claim of an exact or fully validated DUV instrument is made.
