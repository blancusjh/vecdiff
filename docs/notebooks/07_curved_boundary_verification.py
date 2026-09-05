# %% [markdown]
# # Curved-interface reflection and refraction: does the propagated field close the boundary?
#
# **Use case:** decide whether the main per-wavevector Fresnel construction is
# sufficiently accurate for a particular curved dielectric interface. Checking
# the algebraic Fresnel traces alone cannot answer this question. We reconstruct
# both outgoing fields with the full dyadic Green kernel and approach the surface
# from the two physical media.
#
# For a source-free dielectric boundary, using $\mathcal{H}=Z_0H_{\rm SI}$,
# $$\hat n\times(E_i+E_r-E_t)=0,\qquad
# \hat n\times(\mathcal{H}_i+\mathcal{H}_r-\mathcal{H}_t)=0,$$
# $$\hat n\cdot[n_1^2(E_i+E_r)-n_2^2E_t]=0,\qquad
# \hat n\cdot(\mathcal{H}_i+\mathcal{H}_r-\mathcal{H}_t)=0.$$
# We test these **after propagation**, retaining reactive near fields. No fitted
# amplitude, fitted phase, auxiliary-source solution, or reference boundary data
# enter the calculation. A finite open aperture is not a complete dielectric body.
# %%
from pathlib import Path
import sys, json
root=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'pyproject.toml').exists())
sys.path.insert(0,str(root))
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','inline')
import numpy as np
import matplotlib.pyplot as plt
from examples.notebook_tools import style,show,scalar_map
from vecdiff import SphericalCap,Medium,DielectricInterface,plane_wave,sample_surface,interface_transform
style()
# %% [markdown]
# ## 1. See the two fields and the actual geometry
# A spherical cap has $R=2\lambda_0$, aperture radius $\lambda_0$, and indices
# $1\to1.5$. The incident plane wave travels in $+z$ with $x$ polarization.
# Here all coordinates are in units of the vacuum wavelength. Below the cap the
# physical field is incident plus reflected; above it the field is transmitted.
# The white band excludes points within $0.05\lambda_0$ vertically of the surface:
# ordinary tensor quadrature is not used to infer a singular boundary limit.
# The next section uses target-centred near-field quadrature for that purpose.
# %%
surface=SphericalCap(2.)
samples=sample_surface(surface,(0,1),(0,2*np.pi),64,128)
incident=plane_wave()
out=interface_transform(incident,DielectricInterface(surface,Medium(),Medium(1.5)),samples)
x=np.linspace(-.8,.8,101); z=np.linspace(-.35,.75,121)
X,Z=np.meshgrid(x,z); points=np.stack((X,0*X,Z),axis=-1)
sag=surface.sag(abs(X)); below=Z<sag-.05; above=Z>sag+.05
physical=np.full(points.shape,np.nan+0j); reflected=np.full(points.shape,np.nan+0j)
ei,hi=incident.evaluate(points[below]); er,hr=out.reflected.evaluate(points[below],chunk=8)
et,ht=out.transmitted.evaluate(points[above],chunk=8)
physical[below]=ei+er; physical[above]=et; reflected[below]=er
fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
for ax,values,title in zip(axes,[np.sum(abs(physical)**2,axis=-1),abs(physical[...,2])**2,np.sum(abs(reflected)**2,axis=-1)],
                           ['Total physical electric field','Longitudinal electric component','Reflected field below the cap']):
    scalar_map(fig,ax,values,x,z,title,xlabel=r'$x/\lambda_0$',ylabel=r'$z/\lambda_0$',
               label=r'$|E|^2/|E_0|^2$' if ax!=axes[1] else r'$|E_z|^2/|E_0|^2$')
    ax.plot(x,surface.sag(abs(x)),color='cyan',lw=1.5); ax.set_aspect('equal')
show(fig,'07_curved_fields')
# %% [markdown]
# ## 2. Separate numerical convergence from physical agreement
# The verification samples $q\pm\delta\hat n$ with
# $\delta/\lambda_0=0.008,0.004,0.002,0.001$. Each source quadrature is held fixed
# through the offset sequence; both radial order and azimuthal resolution are
# independently increased. Quadratic extrapolation of the **complex fields** is
# $$F(0)\simeq\tfrac13F(\delta)-2F(\delta/2)+\tfrac83F(\delta/4).$$
# Comparing overlapping triples checks sensitivity to the boundary offset.
# Tangential $E$ is normalized by incident $E_0$, tangential $\mathcal{H}$ and normal
# $B$ by $n_1E_0$, and normal $D$ by $\max(n_1^2,n_2^2)E_0$. These are amplitude
# residuals, not percentages of transmitted power. The probe has
# $\rho/R=0.2$, $\phi=45^\circ$; this is a necessary pointwise test, not a proof
# over the entire surface.
# %%
from benchmarks.curved_boundary_limits import case
fresh=case(10.,1.,1.5,illumination='gaussian')
report=json.loads((root/'benchmarks/results/curved_boundary_limits.json').read_text())
rows=report['cases']; saved=rows[-1]
assert fresh['numerical_convergence_passed']
np.testing.assert_allclose(list(fresh['extrapolated_boundary_residuals'].values()),
                           list(saved['extrapolated_boundary_residuals'].values()),rtol=2e-5,atol=2e-8)
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for key,label in [('tangential_E',r'$E_t$'),('tangential_H',r'$\mathcal{H}_t$'),('normal_D',r'$D_n$'),('normal_B',r'$B_n$')]:
    axes[0].loglog(fresh['offsets_over_wavelength'],[r[key] for r in fresh['offset_boundary_residuals']],'o-',label=label)
    axes[0].axhline(fresh['extrapolated_boundary_residuals'][key],lw=.7,alpha=.5)
for indices in [[1.,1.5],[1.5,1.]]:
    subset=[r for r in rows if r['indices']==indices and r['illumination']=='plane' and not r['flat_control'] and r['incidence_degrees']==0]
    axes[1].loglog([r['radius_over_wavelength'] for r in subset],
                  [max(r['extrapolated_boundary_residuals'].values()) for r in subset],'o-',label=f'{indices[0]} → {indices[1]}')
axes[0].set(xlabel=r'$\delta/\lambda_0$',ylabel='Normalized boundary mismatch',title='Localized beam: approach from both sides')
axes[1].axhline(.01,color='k',ls='--',label='1% amplitude criterion')
axes[1].set(xlabel=r'$R/\lambda_0$',ylabel='Maximum of four residuals',title='Hard aperture: size alone is not validation')
for ax in axes: ax.legend(); ax.grid(alpha=.2)
show(fig,'07_boundary_convergence')
print(f"Quadrature change: {fresh['max_quadrature_change']:.3e}; extrapolation change: {fresh['boundary_extrapolation_change']:.3e}")
# %% [markdown]
# ## 3. Aperture and equal-index controls
# The localized illumination is an explicitly transverse, 121-mode Maxwell
# spectrum obtained from an 11×11 Gauss–Hermite Gaussian quadrature. Its nominal
# waist is $a/2=2.5\lambda_0$. This finite spectrum is the defined input;
# convergence to an ideal Gaussian is not part of the boundary claim.
#
# Hard truncation radiates even when both indices are equal. A flat, truncated
# aperture is also not the exact infinite-plane Fresnel problem of notebook 02.
# Consequently a curved hard-aperture residual cannot be assigned entirely to
# curvature. Localizing the incident field reduces the equal-index mismatch,
# and the genuine dielectric test can be interpreted against that control.
# %%
controls=[rows[2],rows[6],rows[7],rows[-2],rows[-1]]
labels=['Curved\nplane wave','Equal index\nplane wave','Flat\nplane wave','Equal index\nlocalized','Curved\nlocalized']
fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
positions=np.arange(len(controls))
for j,key in enumerate(['tangential_E','tangential_H','normal_D','normal_B']):
    ax.bar(positions+(j-1.5)*.18,[100*r['extrapolated_boundary_residuals'][key] for r in controls],width=.18,label=key.replace('_',' '))
ax.set_xticks(positions,labels); ax.set(ylabel='Normalized amplitude mismatch (%)',title=r'Controls at $R=10\lambda_0$, aperture $5\lambda_0$')
ax.axhline(1,color='k',ls='--',lw=1); ax.legend(ncol=4,fontsize=9)
show(fig,'07_boundary_controls')
for label,row in zip(labels,controls):
    print(label.replace('\n',' '),f"max residual={100*max(row['extrapolated_boundary_residuals'].values()):.4f}%",
          'numerically converged:',row['numerical_convergence_passed'])
# %% [markdown]
# **Decision:** the localized $R=10\lambda_0$ test meets the stated 1% amplitude
# criterion at this probe, while the hard-aperture examples do not. Neither result
# establishes exact dielectric scattering, closed-sphere resonances, or uniform
# error over a complete lens. The benchmark deliberately records physical failure
# alongside numerical success. Additional aperture, illumination, location, and
# curvature tests are required before extending this acceptance domain.
# %% [markdown]
# ## 4. Move the boundary probe toward the aperture edge
# A pointwise pass near the beam centre is not uniform acceptance. This scan
# holds $R=10\lambda_0$, the incident finite spectrum, and aperture fixed, and
# varies the probe radius along $\phi=45^\circ$. The second panel divides each
# residual by $\|E_i(q)\|/E_0$; its magnetic reference is $n_1\|E_i(q)\|$.
# It exposes errors that appear small only because the local illumination is weak.
# The equal-index control undergoes the identical reconstruction.
# %%
scan=report['localized_probe_scan']
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for n2,label in [(1.,'Equal-index control'),(1.5,'Curved dielectric')]:
    subset=[r for r in scan if r['indices'][1]==n2]
    assert all(r['numerical_convergence_passed'] for r in subset)
    for ax,key in zip(axes,['extrapolated_boundary_residuals','local_amplitude_normalized_boundary_residuals']):
        ax.semilogy([r['probe_radius_over_radius'] for r in subset],[100*max(r[key].values()) for r in subset],'o-',label=label)
for ax in axes:
    ax.axhline(1,color='k',ls='--',lw=1); ax.set(xlabel=r'Probe radius $\rho/R$',ylabel='Largest normalized amplitude mismatch (%)'); ax.legend(); ax.grid(alpha=.2)
axes[0].set_title('Normalization by incident beam peak')
axes[1].set_title('Normalization by local incident amplitude')
show(fig,'07_boundary_position_scan')
for row in scan:
    if row['indices'][1]==1.5:
        print(f"ρ/R={row['probe_radius_over_radius']:.1f}: incident |E|={row['incident_electric_norm_at_probe']:.4f}, local normalized residual={100*max(row['local_amplitude_normalized_boundary_residuals'].values()):.3f}%")
