# %% [markdown]
# # A glass–air interface: beam splitting, Brewster incidence and total internal reflection
# 
# **Experiment.** Send a monochromatic, finite-width beam towards a plane at $z=0$.
# Recover both outgoing fields, identify the critical angle, and inspect the field
# that remains in air under total internal reflection (TIR).
# 
# All distances below are in µm; $\lambda_0=0.532\,\mathrm{µm}$, $n_1=1.5$, $n_2=1$.
# Phasors use $e^{-i\omega t}$. The incident peak electric amplitude is 1 V/m.
# The beam is uniform along $y$ (a 2D beam cross-section, not a finite-power 3D Gaussian).
# Each angular component is transformed with the production `interface_transform`.
# Scalar maps show $|\mathbf E|^2$, or the stated component, using **`cmap="hot"`**;
# these are electric-field norms, not automatically normal power flux.
# 
# The layout is inspired by [Diffractio's reflection/refraction example](https://diffractio.readthedocs.io/en/latest/source/examples_scalar/reflection_refraction.html).
# The calculation here uses vecdiff's vector Maxwell spectra, retaining evanescent transmission.
# %%
from pathlib import Path
import sys
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(root))
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")
import numpy as np
import matplotlib.pyplot as plt
from examples.notebook_tools import style, show, scalar_map, polarization_map, fwhm
style()
# %% [markdown]
# ## 1. Predict the classical angles, then measure the transformed waves
# Tangential phase matching gives $n_1\sin\theta_i=n_2\sin\theta_t$.
# For nonmagnetic lossless media,
# $$\theta_B=\tan^{-1}(n_2/n_1),\qquad \theta_c=\sin^{-1}(n_2/n_1).$$
# Brewster incidence suppresses **p** reflection. Above $\theta_c$, $k_{tz}=i\kappa$
# with $\kappa=k_0\sqrt{n_1^2\sin^2\theta_i-n_2^2}>0$.
# The transmitted field decays as $e^{-\kappa z}$; zero transmitted normal power
# does not mean zero transmitted electric field.
# %%
from vecdiff import Medium, Plane, DielectricInterface, ElectricSpectrum, plane_wave, interface_transform
from vecdiff.observables.electromagnetism import boundary_residuals, poynting
wavelength = .532
n1, n2 = 1.5, 1.
interface = DielectricInterface(Plane(), Medium(n1), Medium(n2))
theta_B = np.degrees(np.arctan(n2/n1))
theta_c = np.degrees(np.arcsin(n2/n1))
print(f"Brewster angle: {theta_B:.6f}°; critical angle: {theta_c:.6f}°")

def incident_wave(angle, polarization):
    t = np.deg2rad(angle)
    e = (0,1,0) if polarization == 's' else (np.cos(t),0,-np.sin(t))
    return plane_wave((np.sin(t),0,np.cos(t)), e, wavelength=wavelength, medium=Medium(n1))

angles = np.unique(np.r_[np.linspace(0,89,600), theta_B, theta_c])
q = np.c_[np.linspace(-2,2,31), np.zeros((31,2))]
curves = {}; jumps = []; snell_errors = []
for pol in ('s','p'):
    R, T = [], []
    for angle in angles:
        wave = incident_wave(angle, pol)
        out = interface_transform(wave, interface)
        ei,hi = wave.evaluate(q); er,hr = out.reflected.evaluate(q); et,ht = out.transmitted.evaluate(q)
        fi = poynting(ei,hi)[:,2].mean()
        R.append(-poynting(er,hr)[:,2].mean()/fi)
        T.append(poynting(et,ht)[:,2].mean()/fi)
        jumps.append(boundary_residuals(ei+er,hi+hr,et,ht,[0,0,1],Medium(n1),Medium(n2),
                      electric_scale=1,magnetic_scale=n1))
        if angle < theta_c:
            kt = out.transmitted.wavevectors[0].real
            measured = np.arctan2(kt[0],kt[2])
            snell_errors.append(abs(n2*np.sin(measured)-n1*np.sin(np.deg2rad(angle))))
    curves[pol] = np.array(R),np.array(T)
fig,axes = plt.subplots(1,2,figsize=(12,4),layout='constrained')
for pol,(R,T) in curves.items():
    axes[0].plot(angles,R,label=f'$R_{pol}$')
    axes[0].plot(angles,T,'--',label=f'$T_{pol}$')
    axes[1].semilogy(angles,np.maximum(abs(R+T-1),1e-17),label=pol)
for ax in axes:
    ax.axvline(theta_B,color='gray',ls=':',label='Brewster' if ax is axes[0] else None)
    ax.axvline(theta_c,color='black',ls=':',label='Critical' if ax is axes[0] else None)
    ax.set(xlabel='Incidence angle (degrees)'); ax.legend(fontsize=9)
axes[0].set(ylabel='Normal power fraction',ylim=(-.02,1.02),title='Reflected and transmitted power')
axes[1].set(ylabel=r'$|R+T-1|$',title='Energy balance of reconstructed E and H')
show(fig,'02_angles_power')
assert max(max(row.values()) for row in jumps) < 2e-12
assert max(snell_errors) < 1e-12
assert curves['p'][0][np.argmin(abs(angles-theta_B))] < 1e-25
assert max(np.max(abs(R+T-1)) for R,T in curves.values()) < 2e-12
print('Maximum normalized boundary jumps:', {key:max(row[key] for row in jumps) for key in jumps[0]})
print(f'Maximum Snell residual: {max(snell_errors):.3g}')
# %% [markdown]
# ## 2. Give the field a finite beam envelope
# We synthesize a Gaussian **angular** amplitude, centered at the chosen incidence:
# $$\mathbf E_i(x,z)=\sum_j w_j\,\mathbf e_j
#  e^{i k_1(x\sin\theta_j+z\cos\theta_j)},\qquad
# w_j\propto e^{-(k_1w_0\,\delta\theta_j)^2/4}.$$
# Here $w_0=2$ µm and the sum spans six angular standard scales. Each s/p basis
# is transverse to its own wavevector. The finite angular width rounds the
# critical transition: a beam centered exactly at $\theta_c$ contains both
# propagating and evanescent transmitted modes. The plane-wave identities above
# remain exact; a finite beam has no single refracted angle at this transition.
# %%
def beam(angle, pol='s', count=161, waist=2.):
    k1 = 2*np.pi*n1/wavelength
    delta = np.linspace(-6/(k1*waist),6/(k1*waist),count)
    theta = np.deg2rad(angle)+delta
    assert np.all((theta > -np.pi/2)&(theta < np.pi/2))
    weights = np.exp(-(k1*waist*delta)**2/4); weights /= weights.sum()
    directions = np.c_[np.sin(theta),0*theta,np.cos(theta)]
    polarization = np.tile([0,1,0],(count,1)) if pol=='s' else np.c_[np.cos(theta),0*theta,-np.sin(theta)]
    return ElectricSpectrum(k1*directions, weights[:,None]*polarization,wavelength,Medium(n1))

x = np.linspace(-12,12,601); z = np.linspace(-8,8,481)
X,Z = np.meshgrid(x,z); points = np.stack((X,0*X,Z),axis=-1)
lower = Z < 0
cases = [(25.,'s','Partial reflection'),(theta_B,'p','Brewster-centred beam'),
         (theta_c,'s','Critical-centred beam'),(55.,'s','Total internal reflection')]
beam_results = []
fig,axes = plt.subplots(2,2,figsize=(13,9),layout='constrained')
for ax,(angle,pol,title) in zip(axes.flat,cases):
    incoming = beam(angle,pol); out = interface_transform(incoming,interface)
    e = np.empty(points.shape,complex)
    ei,hi = incoming.evaluate(points[lower]); er,hr = out.reflected.evaluate(points[lower])
    et,ht = out.transmitted.evaluate(points[~lower])  # Never continue growing evanescent waves into z<0.
    e[lower] = ei+er; e[~lower] = et
    scalar_map(fig,ax,np.sum(abs(e)**2,axis=-1),x,z,f'{title}: {angle:.2f}°, {pol}',vmax=4)
    ax.axhline(0,color='cyan',lw=1); ax.text(-11,6.6,'air: n=1',color='white')
    ax.text(-11,-7,'glass: n=1.5',color='white'); ax.set_aspect('equal')
    beam_results.append((incoming,out,e))
show(fig,'02_beam_refraction_tir')
# %% [markdown]
# ## 3. Separate the three branches
# The previous maps show the **coherent sum** below the interface. Interference
# there is physical. Separate branch norms clarify the incoming and outgoing
# beam directions. All four panels below use the same incident amplitude scale;
# none is independently peak-normalized.
# %%
incoming,out,e = beam_results[-1]
separate = []
for spectrum,region in [(incoming,lower),(out.reflected,lower),(out.transmitted,~lower)]:
    values = np.full(X.shape,np.nan)
    values[region] = np.sum(abs(spectrum.evaluate(points[region])[0])**2,axis=-1)
    separate.append(values)
fig,axes = plt.subplots(1,4,figsize=(16,4.5),layout='constrained')
for ax,values,title in zip(axes,[*separate,np.sum(abs(e)**2,axis=-1)],
                            ['Incident','Reflected','Transmitted / evanescent','Physical total']):
    scalar_map(fig,ax,values,x,z,title,vmax=4)
    ax.axhline(0,color='cyan',lw=.8); ax.set_aspect('equal')
show(fig,'02_separate_branches')
# %% [markdown]
# ## 4. Resolve the evanescent penetration and test the finite beam
# For a single s-polarized plane wave at 55°, the **amplitude** penetration depth
# is $\delta_E=1/\kappa$ and the intensity depth is $\delta_I=1/(2\kappa)$.
# We compare the computed decay to this formula, and evaluate all four boundary
# conditions after summing the finite-beam spectra at 301 interface positions.
# %%
wave = incident_wave(55,'s'); out = interface_transform(wave,interface)
kappa = (2*np.pi/wavelength)*np.sqrt(n1**2*np.sin(np.deg2rad(55))**2-n2**2)
zz = np.linspace(0,1.2,301)
evanescent,hh = out.transmitted.evaluate(np.c_[0*zz,0*zz,zz])
decay = abs(evanescent[:,1]/evanescent[0,1])**2
fig,axes = plt.subplots(1,2,figsize=(12,4),layout='constrained')
axes[0].semilogy(zz,decay,label='Computed transmitted field')
axes[0].semilogy(zz[::15],np.exp(-2*kappa*zz[::15]),'o',mfc='none',label=r'$e^{-2\kappa z}$')
axes[0].axvline(1/(2*kappa),color='gray',ls=':'); axes[0].legend()
axes[0].set(xlabel='Distance into air (µm)',ylabel=r'$|E_y(z)/E_y(0)|^2$',title='Evanescent intensity penetration')
probe = np.c_[np.linspace(-10,10,301),np.zeros((301,2))]
finite_jumps=[]
for incoming,out,_ in beam_results:
    ei,hi=incoming.evaluate(probe); er,hr=out.reflected.evaluate(probe); et,ht=out.transmitted.evaluate(probe)
    finite_jumps.append(boundary_residuals(ei+er,hi+hr,et,ht,[0,0,1],Medium(n1),Medium(n2),electric_scale=1,magnetic_scale=n1))
for i,key in enumerate(finite_jumps[0]):
    axes[1].semilogy(range(4),[max(row[key],1e-18) for row in finite_jumps],'o-',label=key)
axes[1].set(xticks=range(4),xticklabels=['Partial','Brewster','Critical','TIR'],ylabel='Normalized RMS jump',title='Finite-beam Maxwell boundary conditions')
axes[1].legend(fontsize=9)
show(fig,'02_evanescent_boundary')
assert np.max(abs(decay-np.exp(-2*kappa*zz))) < 1e-12
assert max(max(row.values()) for row in finite_jumps) < 2e-12
print(f'Amplitude depth = {1/kappa*1e3:.2f} nm; intensity depth = {1/(2*kappa)*1e3:.2f} nm')
# A spectral refinement check on complex E, including points in both media.
probe = points[::60,::75].reshape(-1,3)
a,b = beam(55,count=161),beam(55,count=321)
def physical_field(wave,probe):
    out=interface_transform(wave,interface); e=np.empty_like(probe,dtype=complex); mask=probe[:,2]<0
    e[mask]=wave.evaluate(probe[mask])[0]+out.reflected.evaluate(probe[mask])[0]
    e[~mask]=out.transmitted.evaluate(probe[~mask])[0]
    return e
coarse,fine=physical_field(a,probe),physical_field(b,probe)
change=np.linalg.norm(coarse-fine)/np.linalg.norm(fine)
print(f'161 → 321 angular nodes: relative complex-field change {change:.3g}')
assert change < 2e-4
# %% [markdown]
# ## 5. Magnify the total-internal-reflection near field
# Resolve the air-side penetration with 3.3 nm axial spacing. The left panel
# shows the finite-beam electric norm; the right is a snapshot of its real
# s-polarized component. Both are scalar field maps with `cmap="hot"`.
# The interface is at z=0; evanescent transmission is visibly nonzero above it.
# %%
zoom_x=np.linspace(-3,3,501); zoom_z=np.linspace(-.5,.7,361)
ZX,ZZ=np.meshgrid(zoom_x,zoom_z)
zoom_points=np.stack((ZX,0*ZX,ZZ),axis=-1)
zoom_e=physical_field(beam(55),zoom_points.reshape(-1,3)).reshape(zoom_points.shape)
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
scalar_map(fig,axes[0],np.sum(abs(zoom_e)**2,axis=-1),zoom_x,zoom_z,
           'TIR near field: electric norm',vmax=4)
scalar_map(fig,axes[1],zoom_e[...,1].real,zoom_x,zoom_z,
           'TIR near field: real Ey at t=0',label='Re Ey / E0',vmin=-2,vmax=2,cmap='hot')
for ax in axes: ax.axhline(0,color='cyan',lw=1)
show(fig,'02_tir_near_field')
# %% [markdown]
# **What this experiment establishes.** The computed fields recover Snell refraction,
# Brewster cancellation, $R+T=1$, TIR and evanescent decay. Maxwell continuity holds
# for the reconstructed finite-beam fields, including its near-critical modes.
# These conclusions concern one infinite, lossless plane. Absorption, roughness,
# finite edges and frustrated TIR across a second nearby interface require their
# own physical configuration.
