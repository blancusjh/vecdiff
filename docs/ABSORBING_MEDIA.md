# Absorbing media: what a complex index changes in the method

Mathematical note for vecdiff. It states what is exact with a complex
refractive index, derives it from Maxwell's equations, gives the measured
evidence, and analyses, for the owner to decide, what light travelling
*through* an absorbing medium along rays would require.

Conventions: time dependence $e^{-i\omega t}$, plane waves
$e^{+i\mathbf{k}\cdot\mathbf{r}}$, nonmagnetic matter ($\mu=\mu_0$). An
absorbing medium has

$$
n=n'+i\kappa,\qquad n'>0,\ \kappa\ge 0,\qquad
\varepsilon=\varepsilon_0 n^2,\quad \operatorname{Im}\varepsilon=2n'\kappa\,\varepsilon_0\ge 0 .
$$

A plane wave $e^{ink_0z}$ then decays as $e^{-\kappa k_0 z}$. Data tabulated
for $e^{+i\omega t}$ ($n'-i\kappa$) are the complex conjugate. Metals have
$n'<1$ and $\kappa$ of a few (silver in the visible roughly
$0.05+3.3i$); $\operatorname{Re}\varepsilon<0$ when $\kappa>n'$. The ideal
lossless plasma $n=i\kappa$ ($n'=0$) is excluded by `media.Medium`.

## 1. Statement

| Case | What | Status |
|---|---|---|
| 1 | Fresnel reflection from an absorbing medium *behind* a surface (`Reflection(surface, medium, behind=<absorbing>)`: a metal mirror, a lossy substrate) | implemented; exact in the same sense as the dielectric reflection |
| 2 | Homogeneous propagation in an absorbing region (angular spectrum, E_z completion, H, `PlaneWaveExpansion`, `PlaneSpectrum`, `PlaneTransport`) | implemented; exact (it was already, up to two comparisons with a complex $k^2$) |
| 3 | Transport along rays *through* an absorbing medium (a `Refraction` into or out of one; any leg whose incident or transmitted medium absorbs) | refused with `media.AbsorbingMediumError`; analysed in section 5 |

The material is `media.Medium(n, name)`; `media.Dielectric` is its lossless
subclass (real $n$), which keeps its meaning and behaviour and, given a
complex $n$, names `Medium`. A metal is not a dielectric, so it is not
called one.

## 2. Case 1: reflection from an absorbing medium

**Claim.** The reflected ray stays in the transparent incident medium. The
map $\Phi$, its Jacobian $J$, the flux factor
$g=\sqrt{|s_{iz}/(s_{rz}J)|}\,e^{-i\pi\kappa_c/2}$ and the caustic index
$\kappa_c$ do not involve the medium behind. Only $r_s,r_p$ do.

**Derivation.** Let a plane wave $\mathbf{e}\,e^{i\mathbf{k}\cdot\mathbf{r}}$,
$\mathbf{k}=n_ik_0(\sin\theta,0,\cos\theta)$, meet $z=0$. Phase matching on
the boundary fixes the tangential wavevector of every wave to the incident
one, $k_x=n_ik_0\sin\theta$ (real). In the medium behind,
$\mathbf{k}_t\cdot\mathbf{k}_t=n_t^2k_0^2$ (Helmholtz, bilinear) gives

$$
k_{tz}=k_0\,w,\qquad w^2=n_t^2-n_i^2\sin^2\theta .
$$

The radiation condition (no source at $z\to+\infty$) requires the field to
decay: $\operatorname{Im}k_{tz}\ge0$. Because
$\operatorname{Im}(n_t^2)=2n'\kappa>0$, $w^2$ lies in the open upper half
plane, so its *principal* root has $\operatorname{Re}w>0$ and
$\operatorname{Im}w>0$: that is the root taken
(`transport.dyadic.absorbing_reflection`). The transmitted wave is
inhomogeneous: its planes of constant phase are normal to
$\operatorname{Re}\mathbf{k}_t=(k_x,0,k_0\operatorname{Re}w)$, its planes
of constant amplitude normal to $\operatorname{Im}\mathbf{k}_t=(0,0,k_0\operatorname{Im}w)$.

Continuity of tangential $\mathbf{E}$ and $\mathbf{H}=\mathbf{k}\times\mathbf{E}/(k_0Z_0)$
(bilinear products: $\mathbf{k}_t$ is complex) gives, with $c_t=w/n_t$,

$$
r_s=\frac{n_ic_i-w}{n_ic_i+w},\qquad
r_p=\frac{n_tc_i-n_ic_t}{n_tc_i+n_ic_t}=\frac{n_t^2c_i-n_iw}{n_t^2c_i+n_iw},
$$

the same expressions as for a dielectric, with the p axes
$\mathbf{p}=\mathbf{s}\times\mathbf{s}_i$, $\mathbf{p}_r=\mathbf{s}\times\mathbf{s}_r$
of the existing dyadic. The second form of $r_p$ avoids forming $c_t$.
Power: $\langle S_z\rangle$ of the transmitted wave at $z=0^+$ is the power
absorbed per unit area, and $R+A=1$ with $R=|r|^2<1$.

Since the factor that carries the field along the reflected ray is the one
of the dielectric case (derived in `transport.rays`), the leg is exact for a
plane wave on a plane, and of leading order (error $\propto1/\text{size}$)
otherwise, exactly as for a dielectric. As $\kappa\to0^+$ the root $w$ tends
to the dielectric one, also beyond the critical angle, where
$w^2<0$ is approached from above and $w\to+i\sqrt{n_i^2\sin^2\theta-n_t^2}$,
the branch `fresnel` takes for total internal reflection: the limit is
continuous. As $|n_t|\to\infty$, $r_s\to-1+2c_i/n_t$ and
$r_p\to1-2/(n_tc_i)$: the ideal mirror ($r_s=-1$, $r_p=1$) with an error
$O(1/|n_t|)$.

**Evidence** (`general/tests/test_absorbing_reflection.py`). Reference: each
plane wave's boundary-value problem solved numerically in the test (4x4
linear system on bases bilinearly transverse to each wave, the decaying root
chosen by $\operatorname{Im}k_z\ge0$), no Fresnel formula.

| Check | $n_t$ | Result |
|---|---|---|
| carried reflected E vs boundary problem, 0..85 deg, s / p / elliptical | 1.5+0.02i, 0.05+3.3i, 2+4i, 1+1e-6i | $\le 7\times10^{-15}$ |
| $R+A=1$, same sweep | same | $<10^{-12}$; $0<R<1$, $A>0$ |
| normal R vs $\lvert(n-1)/(n+1)\rvert^2$ | 0.0401, 0.983, 0.680, 2.5e-13 | $<10^{-14}$ |
| $\kappa\to0$: $\lvert r(n+i\kappa)-r(n)\rvert$, $\kappa=10^{-3}\ldots10^{-9}$, incl. total reflection | 1.5, 1.0 behind | falls $\ge50\times$ per step, $<10^{-7}$ |
| lattice leg + re-expansion, three lattice plane waves | all four | E, H within $10^{-12}$ of the incident amplitude |

Cylindrical beam on a plane mirror against its per-plane-wave sum (the
leading order's error, sizes 1, 2, 4; reference plane 40 size wavelengths
in front, focus behind):

| Medium behind | polarization, incidence | size 1 | size 2 | size 4 |
|---|---|---|---|---|
| 0.05+3.3i (silver) | TE, normal | 9.4 % | 4.8 % | 2.4 % |
| 0.05+3.3i | TM, about 29 deg | 6.7 % | 3.4 % | 1.7 % |
| 2+4i | TM, about 55 deg | 13 % | 6.3 % | 3.2 % |
| 1.5+0.02i (lossy glass) | TM, about 40 deg | 29 % | 15 % | 7.4 % |

The silver TE row equals the ideal mirror's (9.4, 4.8, 2.4 % in
`test_reflection`): the error is the map's, not the coefficients'.

Concave spherical mirror (R = 160 lambda) under a collimated Gaussian beam,
through `Propagator`, field near focus compared with the ideal mirror, for
$n_t=N(0.3+i)$:

| N | 10 | 100 | 1000 |
|---|---|---|---|
| meridian | 18.6 % | 1.91 % | 0.192 % |
| lattice | 18.6 % | 1.91 % | 0.192 % |

($\approx2/|n_t|$; the two realizations agree to $10^{-9}$.) Pseudo-Brewster
angle of the lossy glass 1.5+0.02i on a convex sphere: the carried $R_p$
has its minimum at 56.31 deg, $R_p=1.37\times10^{-5}$, as the boundary
problem's minimum (56.3102 deg, against $\arctan1.5=56.3099$ deg for the
lossless glass).

## 3. Case 2: plane waves in an absorbing region

**Claim.** The angular spectrum of a homogeneous absorbing region is exact
with a real transverse wavevector on the plane and

$$
k_z=\sqrt{n^2k_0^2-\kappa_t^2},\qquad \operatorname{Im}k_z>0,
$$

$\kappa_t$ the transverse wavenumber (not the extinction coefficient),
$E_z=-(\boldsymbol{\kappa}_t\cdot\mathbf{E}_t)/k_z$ and
$\mathbf{H}=\mathbf{k}\times\mathbf{E}/(k_0Z_0)$, all with complex $\mathbf{k}$.

**Derivation.** In a homogeneous medium Maxwell's equations for
$e^{i\mathbf{k}\cdot\mathbf{r}}$ fields are $\mathbf{k}\times\mathbf{E}=k_0Z_0\mathbf{H}$,
$\mathbf{k}\times\mathbf{H}=-n^2k_0\mathbf{E}/Z_0$, hence
$\mathbf{k}\cdot\mathbf{E}=0$ and $\mathbf{k}\cdot\mathbf{k}=n^2k_0^2$. These are
polynomial identities in the components of $\mathbf{k}$; nothing requires
$\mathbf{k}$ real, and the products are bilinear ($\mathbf{k}\cdot\mathbf{E}$,
not $\mathbf{k}\cdot\overline{\mathbf{E}}$). The Fourier transform of a field
given on a plane fixes $\boldsymbol\kappa_t$ real; the dispersion relation
fixes $k_z^2$; the branch is fixed by the field's direction of propagation:
$n^2k_0^2-\kappa_t^2$ has imaginary part $2n'\kappa k_0^2>0$ for every
$\kappa_t$, so the principal root used by `fields.angular_spectrum` has
$\operatorname{Re}k_z>0$, $\operatorname{Im}k_z>0$ (forward and decaying),
and $-\sqrt{\cdot}$ for a backward spectrum. No mode lies on a "light
circle": $k_z\ne0$ for every $\kappa_t$, so the completion is defined for
all modes, and there is no propagating disk to keep: `evanescent='discard'`
is refused in an absorbing region, and the `Propagator` expands a sampled
field there with all its modes.

What changed in the code: the light-circle test and the dispersion check
compared magnitudes with the complex $k^2$ (now $|k|^2$, identical for real
$k$); the Maxwell residuals took a real index only. The paraxial (Fresnel)
model is refused; "asm" is exact. FINUFFT needs real coordinates: in a plane
spectrum $\operatorname{Im}\mathbf{k}$ is along the normal $\boldsymbol\nu$,
so on a parallel target plane $\boldsymbol\nu\cdot\mathbf{r}=c$ the decay
$e^{-\operatorname{Im}k_z c}$ is a per-mode factor folded into the
amplitudes, and the transform runs on $\operatorname{Re}\mathbf{k}$
(`fields.spectrum._modal_sum`); other targets take the exact direct sum.

**Evidence** (`general/tests/test_absorbing_propagation.py`):

| Check | Result |
|---|---|
| plane wave $e^{ink_0z}$, $n\in\{1.5+0.02i,0.05+3.3i,2+4i,1+10^{-6}i\}$: amplitude $e^{-\kappa k_0z}$, phase $n'k_0z$ | exact to 0 (bound $10^{-13}$) |
| $Z_0H/E=n$ (complex) | $3\times10^{-16}$ |
| oblique inhomogeneous wave, TE and TM, $\operatorname{Re}\mathbf{k}$ at 25-87 deg to $\operatorname{Im}\mathbf{k}$ | E, H to $1.2\times10^{-14}$ |
| divergence, Helmholtz, curl-curl residuals with complex $k$ | $<3\times10^{-16}$ |
| Gaussian beam ($w=1.5\lambda$) vs direct quadrature of its plane-wave integral (Hankel orders 0 and 1, $E_x$ and $E_z$): 1.5+1e-3i at 6 lambda, 1.5+0.3i at 1.5 lambda, 0.2+3i at 0.15 lambda | $5.5\times10^{-15}$ of the peak |
| a complete E with $\mathbf{k}\cdot\overline{\mathbf{E}}=0$ instead of $\mathbf{k}\cdot\mathbf{E}=0$ | refused |
| folded FINUFFT vs direct sum on a plane | $<10^{-10}$ |

The beam decays faster than its axial plane wave (its oblique waves travel
farther per unit depth), as the quadrature shows.

## 4. What is refused, and why

Every refusal is `media.AbsorbingMediumError` (a `ValueError`), raised when
the object is constructed (a `Propagator`, a leg, a beam, a reference), with
its reason and a pointer to this note. Light through an absorbing medium in a
transport leg (case 3); beams (`PlaneBeam`, `FocusedBeam`, `BeamSpectrum`),
whose quadratures over their plane waves take a real wavenumber (a field on
a plane goes through `fields.angular_spectrum`, exact); `HankelSpectrum` and
`LatticeSpectrum`, which hold propagating waves sampled in their real polar
angle (they are leg outputs, transparent wherever a leg is supported);
energy, momentum and stress densities, which for absorbing (necessarily
dispersive) matter are not determined by $n$ at one frequency; the
wave-transport alternatives; `MieSphere`, `LayeredSphere`,
`InfiniteCylinder` and the Debye functions, whose kernels take real
arguments; the validation solvers (BEM, FEM, MFS).

**What the exact references need.** Mie with a complex relative index $m$ is
a standard exact solution. `references._bessel` evaluates the Riccati-Bessel
functions and their logarithmic derivatives by recurrences written for real
$x$. A complex $mx$ requires: the logarithmic derivative
$D_n(mx)=\psi_n'(mx)/\psi_n(mx)$ by downward recurrence (stable for complex
arguments; Bohren & Huffman ch. 4, Wiscombe's criterion for the starting
order); interior radial functions $j_n(mkr)$ scaled by
$e^{-|\operatorname{Im}(mkr)|}$ to avoid overflow (for silver
$|\operatorname{Im}mx|\approx3.3x$, and $x$ reaches $10^3$-$10^4$ in the
tests); the Debye-series coefficients from those $D_n$ with complex
Wronskians; and for layered spheres a recursion that is stable when a shell
absorbs (Yang's 2003 scheme is the one I would use). The exterior functions
are unchanged. That is a self-contained piece of work with its own tests
(convergence in order, the lossless limit, the optical theorem with
absorption $Q_\text{ext}=Q_\text{sca}+Q_\text{abs}$), so it is not done here.

## 5. Case 3: transport through an absorbing medium (design analysis)

### 5.1 The transmitted wave at a transparent | absorbing boundary

For a plane wave from $n_i$ (real) the transmitted wavevector is
$\mathbf{k}_t=(k_x,0,k_0w)$ with $k_x=n_ik_0\sin\theta_i$ real and
$w=\sqrt{n^2-n_i^2\sin^2\theta_i}$ in the first quadrant (section 2). Write
$w=p+iq$. The phase normal makes the real angle $\theta'$,

$$
\tan\theta'=\frac{n_i\sin\theta_i}{p},\qquad
N(\theta_i)\sin\theta'=n_i\sin\theta_i,\quad N=\sqrt{n_i^2\sin^2\theta_i+p^2},
$$

the generalised Snell law with an angle-dependent real index $N$ (Born &
Wolf, ch. 14 in the 7th edition, "optics of metals"; Stratton §9.8; from
memory). The amplitude decays along the normal at the rate $k_0q$,
whatever $\theta_i$: the transmitted wave is inhomogeneous, and the
directions of phase and decay differ by $\theta'$. To first order in
$\kappa/n'$, $p=\sqrt{n'^2-n_i^2\sin^2\theta_i}+O(\kappa^2)$ (the real Snell
law of $n'$, so $\theta'$ differs from the $n'$ ray at $O(\kappa^2)$) and
$q=n'\kappa/\cos\theta'+O(\kappa^3)$. For a metal ($n'\ll n_i\sin\theta_i$)
$p$ is not close to $\sqrt{n'^2-\ldots}$, which is imaginary: *real rays of
$n'$ do not exist* beyond $\sin\theta_i=n'/n_i$ (for silver, beyond about 3
degrees), while the exact wave exists for every angle.

### 5.2 Why the eikonal becomes complex

Insert $\mathbf{E}=\mathbf{e}\,e^{ik_0S}$ into Maxwell's equations of the
absorbing region and collect powers of $k_0$, as for a dielectric. Order
$k_0^2$: $\nabla S\cdot\nabla S=n^2$, now with complex right side. A real
$S$ cannot satisfy it; $S=S'+iS''$ gives

$$
|\nabla S'|^2-|\nabla S''|^2=n'^2-\kappa^2,\qquad
\nabla S'\cdot\nabla S''=n'\kappa .
$$

The characteristics of the complex eikonal equation (its rays) are complex:
$d\mathbf{r}/d\tau=\nabla S$ has complex solutions, and the boundary data on
a real surface (the incident phase, real) generate rays that leave real
space. Order $k_0^1$ is the transport equation
$2\nabla S\cdot\nabla\mathbf{e}+\mathbf{e}\nabla^2S=0$, bilinear, with
$\mathbf{e}\cdot\nabla S=0$ bilinearly.

### 5.3 The leading-order asymptotic solution: complex rays

The leg's field on the output plane is, exactly for a plane surface and to
leading order otherwise, a superposition
$\int A(\mathbf{q})\,T(\mathbf{q})\,e^{ik_0\psi(\mathbf{q};\mathbf{x})}d\mathbf{q}$
of the incident plane waves, each transmitted, with
$k_0\psi=\mathbf{q}\cdot\mathbf{x}+k_{zi}(\mathbf{q})(\ldots)+k_{zt}(\mathbf{q})(\ldots)$.
With $k_{zt}$ complex the phase is complex. Its leading-order evaluation is
the saddle-point method: the saddle $\partial\psi/\partial\mathbf{q}=0$ is
complex, $\mathbf{q}_s=\mathbf{q}_0+\delta\mathbf{q}$, and the result is

$$
A(\mathbf{q}_s)\,T(\mathbf{q}_s)\,e^{ik_0\psi(\mathbf{q}_s)}
\sqrt{\frac{(2\pi)^2}{\det(-ik_0\,\partial^2\psi)}}\Big|_{\mathbf{q}_s},
$$

asymptotically exact as $k_0L\to\infty$ at fixed $\kappa$ (Felsen & Marcuvitz;
Kravtsov & Orlov, *Geometrical Optics of Inhomogeneous Media*, on complex
rays; Chapman et al., "Complex ray theory", 1999; from memory). In the ray
language of the leg: the plane-surface-plane map $\Phi$ is continued
analytically, and the preimage $\boldsymbol\rho=\Phi^{-1}(\mathbf{x})$ of a
real output point is complex. "Evaluating the incident field at the
preimage" then means its analytic continuation $\mathbf{E}_i(\boldsymbol\rho)$
to complex positions.

**Can the library's representations be continued?** Its exact
representations are finite plane-wave sums $\sum_j\mathbf{a}_je^{i\mathbf{k}_j\cdot\mathbf{r}}$
(beams on their quadratures, PlaneSpectrum, LatticeSpectrum), entire
functions of $\mathbf{r}$: the direct sum already accepts complex
coordinates. But the continuation multiplies each term by
$e^{-\mathbf{k}_j\cdot\operatorname{Im}\boldsymbol\rho}$, and the sum is then
evaluated with cancellation of order $e^{k|\operatorname{Im}\boldsymbol\rho|}$.
Section 5.4 gives $|\operatorname{Im}\boldsymbol\rho|=O(\kappa L)$, so the
loss of digits is $O(\kappa k_0L)/\ln10$: affordable while light survives
($\kappa k_0L=O(1)$), hopeless for a metal. A sampled field is a plane-wave
sum too, with the same limit. The surfaces must also be continued: the
library's spheres and aspheres are analytic formulas that numpy evaluates
at complex $(x,y)$ (branch of the square root to be fixed continuously),
callable freeforms are not guaranteed to be. Newton inversion of $\Phi$
works in $\mathbb{C}^2$ with the same analytic Jacobian.

**Flux factor and caustic index.** $J=\det D\Phi$ becomes complex and
generically never vanishes on real rays: caustics move off real space. The
root $1/\sqrt{J}$ must be the continuous branch along the (complex) ray
from the input plane, i.e. the KMAH index generalises to the winding of
$\arg J$ along the path ($\kappa_c=$ the number of half-turns of $\arg J$
through $-\pi$, counted as the real count is), and the absolute value in
$g=\sqrt{|s_{iz}c_t/(c_is_{tz}J)|}$ must go: the factor is
$\sqrt{s_{iz}c_t/(c_is_{tz}J)}$ continued, with complex $c_t$ and $s_{tz}$.

### 5.4 The real-ray shortcut and its error term

Shortcut: trace real rays of $n'$; carry along them with
$\Delta S=n_il_i+n\,l_t$ (complex $n$, real legs: phase $n'k_0l_t$ and
attenuation $e^{-\kappa k_0l_t}$); take $t_s,t_p$ at the complex $n$ and the
real incidence; take the tangential projection of the transmitted wave
with the complex $c_t$; keep the flux factor of the real $n'$ map. Expand
the exact phase in $\kappa$ (one plane surface, transmitted depth $z$):

$$
k_{zt}(q)=k'_z+i\kappa\frac{n'k_0^2}{k'_z}+\frac{\kappa^2}{2}\frac{k_0^2q^2}{k_z'^3}+O(\kappa^3),
\qquad k'_z=\sqrt{n'^2k_0^2-q^2}.
$$

The first-order term at the real saddle $q_0$ is $i\kappa k_0z/\cos\theta'=i\kappa k_0l_t$:
the shortcut's attenuation. What it leaves out:

1. the second-order phase
   $\delta\phi=\tfrac12\kappa^2k_0z\,\sin^2\theta'/(n'\cos^3\theta')$, which
   *grows with the size*: relative error $O(\kappa^2k_0L)$;
2. the shift of the saddle, $\delta q=-i\kappa\,\alpha'(q_0)z/\psi''(q_0)$
   with $\alpha=n'k_0^2/k'_z$, of order $\kappa k_0$ (size-independent):
   the amplitude $A(q_s)T(q_s)$ differs from $A(q_0)T(q_0)$ by
   $O(\kappa/\mathrm{NA})$ for a beam whose angular spectrum varies on the
   scale of its numerical aperture; completing the square gives the
   phase term of item 1 again.

So at fixed $\kappa$ the shortcut does *not* tend to the exact field as the
system grows: its error $O(\kappa^2k_0L)+O(\kappa/\mathrm{NA})$ stays or
grows while the method's own error falls as $1/(k_0L)$. But light that
crosses an absorbing thickness $L$ survives only if $\kappa k_0L=O(1)$. In
that regime $\kappa=O(1/(k_0L))$ and both terms are $O(1/(k_0L))$: **the
shortcut is consistent at the method's own order exactly where there is
light to carry**, and its error terms say when it is not
($\kappa^2k_0L$ or $\kappa/\mathrm{NA}$ not small against $1/(k_0L)$), which
is checkable per leg from the rays (the leg knows $l_t$ and $\theta'$). It is
undefined for a metal beyond $\sin\theta_i=n'/n_i$ (5.1), where there is
nothing to carry anyway: the field dies within the skin depth
$1/(k_0\kappa)\approx\lambda/20$.

**The reference plane matters.** The leg returns to the *same* plane, so in
the transmitted medium $l_t$ is negative when the plane lies before the
surface: the carried field is the transmitted one continued backwards,
amplified by $e^{\kappa k_0|l_t|}$, and its re-expansion then decays
forward: cancellation of $e^{2\kappa k_0|l_t|}$. With the plane at the
boundary, $l_t$ is bounded by the sag of the surface, and the bulk
absorption is carried exactly by case 2.

**Prototype numbers** (scratch only, not in the library:
`absorb_scratch/case3_plane.py`). Cylindrical beam (NA 0.3, focus 80 size
wavelengths behind) from air onto a plane, the incident light moved exactly
to the boundary plane, carried there along real $n'$ rays with complex
$t_s,t_p$ and complex $c_t$, compared with the per-plane-wave
boundary-value solution at $z=0^+$:

| $n$ | polarization, incidence | size 1 | size 2 | size 4 |
|---|---|---|---|---|
| 1.5+0.02i | TE, normal | 0.039 % | 0.020 % | 0.0099 % |
| 1.5+0.02i | TM, about 29 deg | 0.040 % | 0.020 % | 0.010 % |
| 1.5+0.3i | TE, normal | 0.045 % | 0.022 % | 0.011 % |
| 1.5+0.3i | TM, about 29 deg | 0.072 % | 0.036 % | 0.018 % |
| 0.05+3.3i | TE, normal | 0.089 % | 0.045 % | 0.023 % |
| 0.05+3.3i | TM, about 29 deg | no real rays | | |
| 1.5+0.3i, real $\cos\theta'$ in the TM projection | TM | 2.43 % | 2.41 % | 2.40 % |

The last row is the $O(\kappa)$ error of item 2 in its simplest form (the
tangential projection of the inhomogeneous wave): it does not fall with the
size. The plane-interface case has $l_t=0$; a curved surface adds the
$O(\kappa^2k_0h)$ term over its sag $h$, not measured here.

### 5.5 Exact solutions that could validate each option

- Absorbing half-space or slab (plane boundaries): per-plane-wave Airy sums,
  the same boundary-value solve as the tests here and in `test_paths`;
  cheap, exact, any $\kappa$. Validates the plane case and multiple passes.
- Absorbing sphere: Mie with complex $m$ (section 4), the Debye terms for
  transmission ($p=1$) and internal reflections ($p\ge2$). Validates a
  curved entrance and exit, where $l_t\ne0$ and the sag matters.
- Absorbing layered sphere (lossy coating on glass, or glass on metal):
  the layered recursion with complex shells.

### 5.6 Recommendation

1. Keep case 3 refused (as now) until the owner decides.
2. If wanted, implement the real-ray shortcut for **weak absorption only**,
   with the reference plane at or beyond the surface in the absorbing
   medium, complex $\Delta S$, complex $t$ and $c_t$, and a per-leg refusal
   when $\kappa^2k_0\,l_\text{max}$ or $\kappa/\mathrm{NA}$ exceeds the
   method's own error level, or when real $n'$ rays do not exist. Cost: a
   transmitted-branch variant of `fresnel`/`carried`, the attenuation in
   `rays`, the guard, and the slab and Mie-$m$ references (the latter
   being the larger part).
3. For metals, transport into the bulk is not worth building: only
   reflection (case 1) carries light; the absorbed power is
   $1-R$, reportable from the coefficients.
4. Complex rays are the asymptotically exact option at any $\kappa$, but
   they need complex Newton inversion, continued surfaces and a winding
   caustic index, and the continuation of the incident field loses
   $\kappa k_0L/\ln10$ digits; I would not build them for this library.

## References

1. M. Born and E. Wolf, *Principles of Optics*, 7th ed. (Cambridge University
   Press, 1999), ch. 14 (optics of metals: the inhomogeneous transmitted wave,
   the generalised Snell law, the pseudo-Brewster angle).
2. J. A. Stratton, *Electromagnetic Theory* (McGraw-Hill, 1941), §9.8-9.9
   (plane waves at the boundary of a conducting medium).
3. C. F. Bohren and D. R. Huffman, *Absorption and Scattering of Light by
   Small Particles* (Wiley, 1983), ch. 4 (Mie theory with complex index, the
   logarithmic derivative by downward recurrence).
4. W. J. Wiscombe, "Improved Mie scattering algorithms," *Appl. Opt.* **19**,
   1505-1509 (1980).
5. W. Yang, "Improved recursive algorithm for light scattering by a
   multilayered sphere," *Appl. Opt.* **42**, 1710-1720 (2003).
6. L. B. Felsen and N. Marcuvitz, *Radiation and Scattering of Waves*
   (Prentice-Hall, 1973), on saddle-point evaluation and complex rays.
7. Yu. A. Kravtsov and Yu. I. Orlov, *Geometrical Optics of Inhomogeneous
   Media* (Springer, 1990), on complex rays and complex eikonals.
8. S. J. Chapman, J. M. H. Lawry, J. R. Ockendon and R. H. Tew, "On the
   theory of complex rays," *SIAM Rev.* **41**, 417-509 (1999).

The bibliographic details, including section numbers, were written from
memory; check them before citing.
