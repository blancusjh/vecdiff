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
# # 07 — Objeto extendido no simétrico: un circuito en litografía hiper-NA
#
# Los cuadernos 01–06 estudiaron la PSF y uno o dos puntos. Aquí cerramos el
# arco con un **objeto extendido sin ninguna simetría**: un patrón de circuito
# (máscara binaria `circuit_pattern.png`) impreso por un sistema de
# **proyección de inmersión de alta apertura**, con reducción $4\times$, en el
# régimen de litografía ArF. La pregunta es la de toda la investigación, ahora
# en su forma más aplicada: **¿qué features resuelve el modelo escalar que el
# vectorial no imprime?**
#
# ## Sistema de proyección
#
# La pupila es la superficie estigmática de salida del sistema: un último
# elemento de LuAG ($n = 2.14$ a 193 nm) hacia agua de inmersión
# ($n = 1.437$). Una salida de sílice a aire no alcanza este régimen — una
# superficie estigmática de denso a raro choca con el ángulo crítico y satura
# cerca de $\mathrm{NA} = 0.74$.
#
# | elemento | parámetros |
# | --- | --- |
# | longitud de onda | $\lambda = 193$ nm (ArF) |
# | superficie de salida | LuAG $n_0 = 2.14 \to$ agua $n_i = 1.437$, $z_0 = -42$ mm, $z_i = +2$ mm |
# | apertura | la útil de la superficie (rasancia): $r = 6.28$ mm $\Rightarrow \mathrm{NA} = 0.883$ |
# | reducción | $4\times$: las trazas de la máscara son $4$ veces la imagen |
# | escala de la imagen | paso fino del circuito $= 1.02\times$ el paso de corte coherente $\lambda/\mathrm{NA}$ |
# | iluminación | Köhler parcialmente coherente, $\sigma = 0.2$ (método de Abbe) |
#
# El modelo de pupila es invariante ante traslación: la imagen se calcula en
# coordenadas de imagen y la reducción es una re-etiqueta exacta de la escala
# de máscara ($x_{\rm másc} = 4\,x_{\rm img}$), sin aberración de campo. El
# paso fino se coloca **en el corte coherente**, que es donde el efecto
# vectorial es máximo: los órdenes $\pm 1$ salen a $\pm\theta$ con
# $\sin\theta = \lambda/(n\,p) \approx \mathrm{NA}/n$, y en TM sus campos
# eléctricos se proyectan con $\cos 2\theta = 0.245$.
#
# ## Iluminación y modos
#
# La máscara es una transmisión binaria $T(x,y)\in\{0,1\}$ con iluminación
# Köhler de coherencia parcial $\sigma = 0.2$: cada punto de la fuente aporta
# una onda plana inclinada, su imagen coherente se forma con la misma pupila,
# y las intensidades se suman en potencia (método de Abbe). La coherencia
# parcial es lo que usa un proyector real: suprime el anillado de bordes que
# con iluminación coherente hace oscilar el contraste de un layout denso.
#
# Los tres modos de pupila:
#
# - **escalar (lente ideal)**: pupila circular sin apodizar, $t_p = t_s = 1$.
#   No resuelve ninguna interfaz: es la referencia de resolución de libro.
# - **escalar físico (canal s)**: la solución exacta del **problema escalar de
#   contorno** de la refracción — Helmholtz en ambos medios con $U$ y
#   $\partial_n U$ continuas, cuya transmisión local es exactamente $t_s$ y
#   cuyo transporte de energía da el mismo $A(Q)$. Verificado contra la
#   integral de Helmholtz–Kirchhoff sobre la superficie real:
#   $|U|/\mathrm{ref} = 1.000000$, RMS del perfil $1.6\times 10^{-8}$.
# - **vectorial**: el operador de transferencia completo (Ecs. 62–63), con
#   canal longitudinal $E_z$ incluido en la suma de Abbe.
#
# En litografía de inmersión hiper-NA este efecto es célebre: el contraste de
# los órdenes TM se degrada como el coseno del ángulo mutuo y obliga a
# iluminación polarizada [D. G. Flagello, T. Milster y A. E. Rosenbluth,
# *JOSA A* **13**, 53 (1996); J. Ruoff y M. Totzeck, *J. Micro/Nanolithogr.
# MEMS MOEMS* **8**, 031404 (2009)].

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

from references.legacy.vecdiff import Grid, FieldCartesian, CartesianSurface
from references.legacy.vecdiff.propagation import transfer_weights_on_grid
from references.legacy.vecdiff.polarization_visualization import plot_field_polarization

π = np.pi
out_dir = Path("output")
out_dir.mkdir(exist_ok=True)

lam = 193e-6  # mm

# Superficie de salida de inmersión: último elemento de LuAG (n ≈ 2.14 a
# 193 nm) hacia agua (n ≈ 1.437).  Una salida de sílice a aire no llega aquí:
# una superficie estigmática de denso a raro choca con el ángulo crítico y
# satura cerca de NA 0.74.
N_LUAG, N_WATER = 2.14, 1.437
LENS = dict(n0=N_LUAG, ni=N_WATER, z0=-42.0, zi=2.0)
SURFACE = CartesianSurface(**LENS)
APERTURE = SURFACE.aperture_limit

# La apertura útil fija el corte coherente.  Nada de esto se postula: sale de
# la geometría de la superficie.
_edge = SURFACE.ray_geometry(np.array([APERTURE * (1.0 - 1e-6)]))
SIN_MAX = float(_edge.sin_ai[0])
NA = LENS["ni"] * SIN_MAX
NU_CUT = NA / lam                 # corte coherente, ciclos/mm
PITCH_CUT = 1.0 / NU_CUT
Kc = 2.0 * π * NA / lam
d_Airy = 2.0 * 3.8317059702075125 / Kc

print(f"superficie de salida: LuAG (n={N_LUAG}) → agua (n={N_WATER})")
print(f"apertura útil r = {APERTURE:.4f} mm,  sin(αi) máx = {SIN_MAX:.4f}")
print(f"NA = {NA:.4f}   d_Airy = {d_Airy / lam:.3f} λ   paso de corte = {PITCH_CUT / lam:.3f} λ")
print(f"k_t/k_z máximo en la apertura: {SIN_MAX / np.sqrt(1 - SIN_MAX**2):.2f}")

# %% [markdown]
# ## Máscara: recorte del patrón de circuito
#
# Recorte de 560×560 px de `circuit_pattern.png` con trazas en ambas
# orientaciones. El PNG original está dibujado a mano y sus bordes son
# dentados; antes de usarlo **rectificamos la máscara**: cada traza se
# reemplaza por un rectángulo perfectamente recto alineado a los ejes (misma
# posición, longitud y ancho medios), de modo que el objeto es un *layout*
# Manhattan ideal y ningún detalle sub-λ espurio contamina el espectro.
# Definimos dos cortes de medición sobre grupos de trazas paralelas con paso
# cercano a la resolución:
#
# - **corte V** (rojo): $y = 3\lambda$, cruza 5 trazas *verticales* — la
#   modulación es a lo largo de $\hat{x}$;
# - **corte H** (azul): $x = -3\lambda$, cruza 3 trazas *horizontales* — la
#   modulación es a lo largo de $\hat{y}$.

# %%
mask_png = (np.asarray(Image.open("../circuit_pattern.png").convert("L"),
                       dtype=float) > 127)
crop_raw = mask_png[40:600, 60:620][::-1]  # eje y hacia arriba (origin="lower")


def rectify_mask(m, min_len=18, w_design=14):
    """Replace each hand-drawn trace by a straight axis-aligned rectangle.

    Median thickness (robust to corner joins) snapped to the layout's design
    width; median centerline; sub-trace slivers dropped; short stubs kept as
    bounding boxes.
    """
    out = np.zeros_like(m)
    for axis in (0, 1):  # 0: barras verticales, 1: horizontales
        kern = np.ones((1, min_len) if axis == 1 else (min_len, 1), bool)
        lab, n = ndimage.label(ndimage.binary_opening(m, structure=kern))
        for k in range(1, n + 1):
            ys, xs = np.nonzero(lab == k)
            run = xs if axis == 1 else ys
            w = int(round(np.median(np.bincount(run - run.min()))))
            if w < 6:                    # astilla en el borde del recorte
                continue
            if abs(w - w_design) <= 6:   # ancho de diseño: 14 px = 0.64 λ
                w = w_design
            c = int(round(np.median(ys if axis == 1 else xs)))
            lo = c - w // 2
            if axis == 1:
                out[lo:lo + w, xs.min():xs.max() + 1] = True
            else:
                out[ys.min():ys.max() + 1, lo:lo + w] = True
    resid = m & ~ndimage.binary_dilation(out, iterations=3)  # muñones cortos
    lab, n = ndimage.label(resid)
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        if len(ys) > 100:
            out[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1] = True
    return out.astype(float)


crop = rectify_mask(crop_raw)
print(f"rectificación: área conservada al {100 * crop.sum() / crop_raw.sum():.1f}%")

pitch_px = 27.0
#: Paso fino del circuito en unidades del paso de corte coherente.  1.02 lo
#: coloca justo donde el efecto TM es máximo; el barrido de diseño da ahí
#: C_escalar = 0.90 contra C_vectorial(TM) = 0.41 con sigma = 0.2.
PITCH_FACTOR = 1.02
#: Reducción del sistema de proyección (lado máscara / lado imagen).
REDUCTION = 4.0
dx = PITCH_FACTOR * PITCH_CUT / pitch_px
#: Los cortes de medición se definieron sobre la máscara a paso 0.95 d_Airy;
#: esta razón los re-escala a la escala actual del patrón.
REL = (PITCH_FACTOR * PITCH_CUT) / (0.95 * d_Airy)
N = 640
pad = (N - crop.shape[0]) // 2
T = np.zeros((N, N))
T[pad:pad + crop.shape[0], pad:pad + crop.shape[1]] = crop

x_ax = (np.arange(N) - N // 2) * dx
grid = Grid.from_axes(x_ax, x_ax)
half_content = 0.5 * crop.shape[0] * dx / lam + 0.3  # ventana útil en λ
print(f"dx = {dx / lam:.4f} λ   contenido = ±{half_content:.1f} λ   "
      f"traza fina = {14 * dx / lam:.2f} λ, paso = {27 * dx / lam:.2f} λ  (imagen)")
print(f"reducción {REDUCTION:.0f}×: en la máscara, traza fina = "
      f"{REDUCTION * 14 * dx / lam:.2f} λ, paso = {REDUCTION * 27 * dx / lam:.2f} λ")

cut_V = dict(axis_value=3.0 * lam * REL, t_min=2.7 * lam * REL, t_max=9.0 * lam * REL, horizontal=True)
cut_H = dict(axis_value=-3.0 * lam * REL, t_min=-11.9 * lam * REL, t_max=-8.4 * lam * REL, horizontal=False)


def mask_trace_centers(cut):
    """Trace centers (in mm) crossed by an axis-aligned cut of the mask."""
    i = np.argmin(np.abs(x_ax - cut["axis_value"]))
    prof = T[i, :] if cut["horizontal"] else T[:, i]
    sel = (x_ax >= cut["t_min"]) & (x_ax <= cut["t_max"])
    b, t = prof[sel], x_ax[sel]
    d = np.diff(np.concatenate(([0.0], b, [0.0])))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return np.array([(t[s] + t[e - 1]) / 2 for s, e in zip(starts, ends)])


centers_V = mask_trace_centers(cut_V)
centers_H = mask_trace_centers(cut_H)
print("centros V (λ):", np.round(centers_V / lam, 2))
print("centros H (λ):", np.round(centers_H / lam, 2))


def draw_cuts(ax, lw=1.4):
    ax.plot([cut_V["t_min"] / lam, cut_V["t_max"] / lam],
            [cut_V["axis_value"] / lam] * 2, "r-", lw=lw)
    ax.plot([cut_H["axis_value"] / lam] * 2,
            [cut_H["t_min"] / lam, cut_H["t_max"] / lam], "-",
            color="dodgerblue", lw=lw)


def show_mask(ax, zoom=None, title="máscara (entrada)"):
    ax.imshow(T, origin="lower", cmap="gray",
              extent=[x_ax[0] / lam, x_ax[-1] / lam, x_ax[0] / lam, x_ax[-1] / lam])
    z = zoom if zoom is not None else (-half_content, half_content,
                                       -half_content, half_content)
    ax.set_xlim(z[0], z[1])
    ax.set_ylim(z[2], z[3])
    ax.set(title=title, xlabel=r"$x/\lambda$", ylabel=r"$y/\lambda$")


fig, ax = plt.subplots(figsize=(8.6, 8.6), constrained_layout=True)
show_mask(ax, title=f"máscara binaria $T(x,y)$ — paso fino = {PITCH_FACTOR:.2f}× el paso de corte"
                   f"  (lado máscara: ×{REDUCTION:.0f})")
draw_cuts(ax, lw=1.8)
ax.text(cut_V["t_max"] / lam + 0.4, cut_V["axis_value"] / lam, "corte V",
        color="r", va="center", fontsize=11)
ax.text(cut_H["axis_value"] / lam, cut_H["t_max"] / lam + 0.5, "corte H",
        color="dodgerblue", ha="center", fontsize=11)
fig.savefig(out_dir / "07_mask_cuts.png", dpi=200)
plt.show()

# %% [markdown]
# ## Propagación por los dos dioptrios
#
# Cuatro propagaciones: escalar ($t_p = t_s = 1$, campo $T\hat{x}$; por
# linealidad e isotropía del modelo escalar la imagen es la misma para
# cualquier polarización de entrada) y vectoriales con $T\hat{x}$, $T\hat{y}$
# y $T(\hat{x}+i\hat{y})/\sqrt{2}$.

# %%
def _radius_from_sine(sin_ai, n_table=100_001):
    """Invertir sin(αi)(r) sobre la superficie: devuelve el radio de pupila r."""
    r_table = np.linspace(0.0, APERTURE * (1.0 - 1e-9), n_table)
    s_table = SURFACE.ray_geometry(r_table).sin_ai
    order = np.argsort(s_table)
    return np.interp(np.clip(sin_ai, 0.0, s_table.max()), s_table[order], r_table[order])


_NUX, _NUY = np.meshgrid(
    np.fft.fftfreq(grid.shape[1], d=float(grid.dx)),
    np.fft.fftfreq(grid.shape[0], d=float(grid.dy)),
    indexing="xy",
)
_NU = np.hypot(_NUX, _NUY)
_PHI = np.arctan2(_NUY, _NUX)
_INSIDE = lam * _NU / LENS["ni"] <= SIN_MAX

_r_pupil = _radius_from_sine(np.where(_INSIDE, lam * _NU / LENS["ni"], 0.0))
_lam_r, _lam_phi, _lam_z = transfer_weights_on_grid(_r_pupil, SURFACE)
_geom = SURFACE.ray_geometry(_r_pupil)
with np.errstate(divide="ignore", invalid="ignore"):
    _jac = np.where(_geom.cos_ai > 0.0, 1.0 / _geom.cos_ai, 0.0)
_zero = np.zeros_like(_lam_r)
W_PLUS = np.where(_INSIDE, 0.5 * (_lam_r + _lam_phi) * _jac, _zero)
W_MINUS = np.where(_INSIDE, 0.5 * (_lam_r - _lam_phi) * _jac, _zero)
W_Z = np.where(_INSIDE, _lam_z * _jac, _zero)
# El canal s es la solución exacta del problema ESCALAR de contorno: Helmholtz
# en ambos medios con U y dU/dn continuas da t_s como transmisión local, y la
# conservación de flujo el mismo A(Q).  Verificado contra la integral de
# Helmholtz-Kirchhoff sobre la superficie: |U|/ref = 1.000000, RMS 1.6e-8.
W_S = np.where(_INSIDE, _lam_phi * _jac, _zero)


def pupil_amplitudes(model):
    """Pesos de pupila de cada modo.

    ``"scalar"``         pupila circular sin apodizar, t_p = t_s = 1: la lente
                         ideal de libro.  No resuelve ninguna interfaz -- es la
                         referencia de resolución, no una solución de contorno.
    ``"scalar_fisico"``  el canal s del operador, A(Q) t_s(theta) con el
                         jacobiano del mapeo: la solución exacta del problema
                         escalar de la refracción (Helmholtz + continuidad de
                         U y dU/dn, que es el problema TE).
    ``"vectorial"``      el operador completo, con canal longitudinal.
    """
    if model == "scalar":
        one = _INSIDE.astype(float)
        return one, np.zeros_like(one), np.zeros_like(one)
    if model == "scalar_fisico":
        return W_S, np.zeros_like(W_MINUS), np.zeros_like(W_Z)
    if model == "vectorial":
        return W_PLUS, W_MINUS, W_Z
    raise ValueError("model debe ser 'scalar', 'scalar_fisico' o 'vectorial'.")


def _source_offsets(sigma, stride=2):
    """Desplazamientos de fuente (Köhler) sobre la malla de frecuencias.

    Se eligen sobre la propia malla del FFT, de modo que desplazar el espectro
    del objeto es un ``np.roll`` exacto y no una interpolación.
    """
    if sigma <= 0.0:
        return [(0, 0, 1.0)]
    dnu_x = abs(_NUX[0, 1] - _NUX[0, 0])
    dnu_y = abs(_NUY[1, 0] - _NUY[0, 0])
    mx = int(np.floor(sigma * NU_CUT / dnu_x))
    my = int(np.floor(sigma * NU_CUT / dnu_y))
    pts = []
    for iy in range(-my, my + 1, stride):
        for ix in range(-mx, mx + 1, stride):
            if (ix * dnu_x) ** 2 + (iy * dnu_y) ** 2 <= (sigma * NU_CUT) ** 2:
                pts.append((iy, ix, 1.0))
    total = sum(w for _, _, w in pts)
    return [(iy, ix, w / total) for iy, ix, w in pts]


#: Grado de coherencia parcial de la iluminación Köhler.  Un sistema de
#: proyección real nunca ilumina de forma coherente: la fuente extendida es lo
#: que suprime el anillado de bordes y estabiliza el contraste.  Con sigma = 0
#: se recupera el caso coherente.
SIGMA = 0.2
_SOURCE = _source_offsets(SIGMA)
print(f"iluminación Köhler: sigma = {SIGMA}, {len(_SOURCE)} puntos de fuente")


def image_intensity(Ex0, Ey0, model="vectorial", sigma=None):
    """Intensidad de imagen por el método de Abbe.

    Cada punto de la fuente ilumina el objeto con una onda plana inclinada,
    se forma su imagen coherente, y las intensidades se suman en potencia.
    """
    src = _SOURCE if sigma is None else _source_offsets(sigma)
    w_plus, w_minus, w_z = pupil_amplitudes(model)
    c2, s2 = np.cos(2.0 * _PHI), np.sin(2.0 * _PHI)

    Sx0 = np.fft.fft2(np.asarray(Ex0, dtype=complex))
    Sy0 = np.fft.fft2(np.asarray(Ey0, dtype=complex))

    I = np.zeros(Sx0.shape, dtype=float)
    Iz = np.zeros_like(I)
    Icross = np.zeros_like(I)
    for iy, ix, w in src:
        Sx = np.roll(Sx0, (iy, ix), axis=(0, 1))
        Sy = np.roll(Sy0, (iy, ix), axis=(0, 1))
        Px = (w_plus + w_minus * c2) * Sx + (w_minus * s2) * Sy
        Py = (w_minus * s2) * Sx + (w_plus - w_minus * c2) * Sy
        Pz = w_z * (np.cos(_PHI) * Sx + np.sin(_PHI) * Sy)
        ex, ey, ez = np.fft.ifft2(Px), np.fft.ifft2(Py), np.fft.ifft2(Pz)
        I += w * (np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2)
        Iz += w * np.abs(ez) ** 2
        Icross += w * np.abs(ey) ** 2
    return I, Iz, Icross


Z0 = np.zeros_like(T)
I_flat, _, _ = image_intensity(T, Z0, "scalar")
I_fis, _, _ = image_intensity(T, Z0, "scalar_fisico")
I_vx, Iz_vx, Icross_vx = image_intensity(T, Z0, "vectorial")
I_vy, Iz_vy, _ = image_intensity(Z0, T, "vectorial")
I_ci, Iz_ci, _ = image_intensity(T / np.sqrt(2.0), 1j * T / np.sqrt(2.0), "vectorial")
xr = grid.X[0, :]
yr = grid.Y[:, 0]


def coherent_fields(Ex0, Ey0):
    """Campos coherentes del punto de fuente axial, con el operador vectorial.

    Las intensidades de arriba son sumas de Abbe y no tienen un campo único;
    los mapas de elipses de polarización y el desglose por canales se dibujan
    sobre la componente coherente axial, que es el punto de fuente dominante.
    """
    w_plus, w_minus, w_z = pupil_amplitudes("vectorial")
    c2, s2 = np.cos(2.0 * _PHI), np.sin(2.0 * _PHI)
    Sx = np.fft.fft2(np.asarray(Ex0, dtype=complex))
    Sy = np.fft.fft2(np.asarray(Ey0, dtype=complex))
    Px = (w_plus + w_minus * c2) * Sx + (w_minus * s2) * Sy
    Py = (w_minus * s2) * Sx + (w_plus - w_minus * c2) * Sy
    Pz = w_z * (np.cos(_PHI) * Sx + np.sin(_PHI) * Sy)
    fld = FieldCartesian(np.fft.ifft2(Px), np.fft.ifft2(Py), grid=grid,
                         symmetric=False)
    return fld, np.fft.ifft2(Pz)


img_vx, Ez_x = coherent_fields(T, Z0)
img_vy, Ez_y = coherent_fields(Z0, T)
img_ci, Ez_c = coherent_fields(T / np.sqrt(2.0), 1j * T / np.sqrt(2.0))

# Dos referencias escalares distintas, y la diferencia importa:
#   I_flat  pupila circular sin apodizar, t_p = t_s = 1 -- la imagen de libro
#           de texto, que no depende de la superficie;
#   I_fis   el canal s (A·t_s·jac): la solución exacta del problema escalar
#           de contorno en la interfaz, verificada a 1.6e-8 contra la
#           integral de Helmholtz-Kirchhoff.
I_esc = I_flat
I_lin = {
    "escalar físico (canal s)": I_fis,
    "vectorial x̂ (transversal)": I_vx - Iz_vx,
    "vectorial x̂ (total)": I_vx,
    "vectorial ŷ (total)": I_vy,
}
I_cir = I_ci

f_cross = float(Icross_vx.sum() / (I_vx - Iz_vx).sum())
f_z = float(Iz_vx.sum() / I_vx.sum())
print(f"iluminación x̂ en el plano imagen:  f_cross = {f_cross:.4f}   f_z = {f_z:.4f}")

# %% [markdown]
# El peso longitudinal en la imagen es $f_z \approx 0.5$: a $\mathrm{NA}=0.94$
# la mitad de la energía del campo enfocado está en $E_z$ — de tres a diez
# veces más que en los sistemas de los cuadernos 05–06 ($f_z \lesssim
# 0.05$–$0.17$). La diatenuación, en cambio, es aquí minúscula
# ($f_{\rm cross} \sim 0.1\%$ para este objeto, cuyo espectro se concentra
# lejos del borde de la pupila): en este régimen **el canal dominante es la
# transversalidad, no Fresnel**.
#
# ### Validación: aumento y PSF del relevo
#
# Medimos el aumento con dos discos y la PSF con un disco casi puntual, y
# verificamos que este sistema compuesto reproduce la fenomenología radial de
# los cuadernos 01–05.

# %%
sep_chk = 6.0 * lam
r_c = 0.15 * d_Airy
X, Y = grid.X, grid.Y
two = (((X - sep_chk / 2) ** 2 + Y ** 2 <= r_c ** 2)
       | ((X + sep_chk / 2) ** 2 + Y ** 2 <= r_c ** 2)).astype(float)
one = ((X ** 2 + Y ** 2) <= r_c ** 2).astype(float)

I2, _, _ = image_intensity(two, Z0, "scalar")
row = I2[np.argmax(I2.max(axis=1))]
n_half = len(xr) // 2
sep_img = (xr[n_half:][np.argmax(row[n_half:])]
           - xr[:n_half][np.argmax(row[:n_half])])
M_meas = sep_img / sep_chk
print(f"|M| medido = {M_meas:.4f}")
# El modelo de pupila es invariante ante traslación, así que la imagen sale en
# las coordenadas del objeto y la magnificación es exactamente uno: no hay
# distorsión que medir, y esta cota solo detecta rupturas gruesas.
assert 0.9 < M_meas < 1.1, M_meas

I_to, Iz_pt, _ = image_intensity(one, Z0, "vectorial")
I_tr = I_to - Iz_pt
I_sc, _, _ = image_intensity(one, Z0, "scalar")


def hwhm_cut(I2d, along_x):
    iy, ix = np.unravel_index(np.argmax(I2d), I2d.shape)
    prof, t = (I2d[iy, :], xr) if along_x else (I2d[:, ix], yr)
    above = np.nonzero(prof >= prof.max() / 2)[0]
    return 0.5 * (t[above[-1]] - t[above[0]])


print(f"PSF escalar:            HWHM = {hwhm_cut(I_sc, True) / lam:.3f} λ")
print(f"PSF vec transversal:    R∥ = {hwhm_cut(I_tr, True) / lam:.3f} λ   "
      f"R⊥ = {hwhm_cut(I_tr, False) / lam:.3f} λ")
print(f"PSF vec total (+|Ez|²): R∥ = {hwhm_cut(I_to, True) / lam:.3f} λ   "
      f"R⊥ = {hwhm_cut(I_to, False) / lam:.3f} λ")

# %% [markdown]
# La jerarquía transversal es la de los cuadernos 01–03
# ($R_\parallel < R_\perp$), y al sumar $|E_z|^2$ la elongación **se
# invierte** ($R_\parallel^{\rm tot} > R_\perp^{\rm tot}$): con $f_z \approx
# 0.5$ estamos del otro lado del cruce $|H_z|^2 = \mathrm{Re}(H_0 H_2^*)$ que
# el cuaderno 05 localizó — el régimen Richards–Wolf pleno.
#
# ## Contraste por orientación: la medición central
#
# Para cada modelo medimos el **contraste de las trazas** sobre los cortes V y
# H: picos en los centros de traza, valles en los puntos medios entre trazas,
# $C = (\bar{I}_{\rm pico} - \bar{I}_{\rm valle})/(\bar{I}_{\rm pico} +
# \bar{I}_{\rm valle})$. La imagen está invertida ($M=-1$): muestreamos en
# $(-x,-y)$ y presentamos todo en las coordenadas de la máscara.

# %%
def image_cut(I2d, cut, n_t=500):
    # El modelo de pupila produce imagen derecha en coordenadas de imagen (la
    # inversión del proyector real es una re-etiqueta global de ejes), así que
    # los cortes muestrean en las mismas coordenadas que la máscara.
    t = np.linspace(cut["t_min"], cut["t_max"], n_t)
    if cut["horizontal"]:
        i = np.argmin(np.abs(yr - cut["axis_value"]))
        vals = np.interp(t, xr, I2d[i, :])
    else:
        i = np.argmin(np.abs(xr - cut["axis_value"]))
        vals = np.interp(t, yr, I2d[:, i])
    return t, vals


def cut_contrast(I2d, cut, centers):
    t, vals = image_cut(I2d, cut)
    peaks = np.interp(centers, t, vals)
    valleys = np.interp(0.5 * (centers[:-1] + centers[1:]), t, vals)
    return (peaks.mean() - valleys.mean()) / (peaks.mean() + valleys.mean())


C = {"escalar": (cut_contrast(I_esc, cut_V, centers_V),
                 cut_contrast(I_esc, cut_H, centers_H))}
C.update({name: (cut_contrast(A, cut_V, centers_V),
                 cut_contrast(A, cut_H, centers_H))
          for name, A in I_lin.items()})
C["circular (total)"] = (cut_contrast(I_cir, cut_V, centers_V),
                         cut_contrast(I_cir, cut_H, centers_H))

w = max(len(k) for k in C)
print("— iluminación lineal —")
print(f"{'modelo':<{w}}   C_V (mod ∥ x̂)   C_H (mod ∥ ŷ)   C_V − C_H")
for name in ["escalar", *I_lin]:
    cv, ch = C[name]
    print(f"{name:<{w}}   {cv:14.3f}   {ch:14.3f}   {cv - ch:+9.3f}")
print("— iluminación circular —")
cv, ch = C["circular (total)"]
print(f"{'circular (total)':<{w}}   {cv:14.3f}   {ch:14.3f}   {cv - ch:+9.3f}")

# %% [markdown]
# La columna $C_V - C_H$ es la **firma de la aberración**, y tiene la virtud
# de no depender de la normalización del modelo escalar. Para el escalar es
# pequeña y de origen puramente geométrico (el grupo V es algo más denso que
# el H: $-0.07$). La iluminación $\hat{x}$ la lleva a $-0.17$ — degrada
# las trazas de modulación paralela mucho más que las perpendiculares —, la
# $\hat{y}$ la invierte a $+0.14$, y la circular la devuelve al nivel
# geométrico repartiendo la pérdida entre ambas orientaciones. La polarización
# **selecciona qué orientación del circuito se degrada**: ningún mecanismo
# del modelo escalar puede producir ese acoplamiento.
#
# ## Iluminación lineal: máscara, escalar y vectorial

# %%
def show_image(ax, A, title, zoom=None, cmap="gray"):
    """Display an image-plane map rotated 180° into mask coordinates."""
    B = A
    ext = [xr[0] / lam, xr[-1] / lam, yr[0] / lam, yr[-1] / lam]
    v = np.percentile(B, 99.6)
    ax.imshow(np.clip(B / v, 0, 1) ** 0.85, origin="lower", extent=ext, cmap=cmap)
    z = zoom if zoom is not None else (-half_content, half_content,
                                       -half_content, half_content)
    ax.set_xlim(z[0], z[1])
    ax.set_ylim(z[2], z[3])
    ax.set(title=title, xlabel=r"$x/\lambda$", ylabel=r"$y/\lambda$")


def annotate_C(ax, name, fontsize=11):
    cv, ch = C[name]
    ax.text(0.02, 0.99, f"$C_V$ = {cv:.2f}\n$C_H$ = {ch:.2f}",
            transform=ax.transAxes, va="top", color="w", fontsize=fontsize)


fig, axes = plt.subplots(2, 2, figsize=(13.6, 14.0), constrained_layout=True)
show_mask(axes[0, 0], title="mask (input, $C_V = C_H = 1$)")
draw_cuts(axes[0, 0])
panels = [(axes[0, 1], I_flat, "escalar puro ($t_p = t_s = 1$, sin apodizar)", "escalar"),
          (axes[1, 0], I_lin["vectorial x̂ (total)"],
           r"vectorial, $\mathbf{E}_0 = T\hat{x}$ — $|E_x|^2+|E_y|^2+|E_z|^2$",
           "vectorial x̂ (total)"),
          (axes[1, 1], I_lin["vectorial ŷ (total)"],
           r"vectorial, $\mathbf{E}_0 = T\hat{y}$ — $|E_x|^2+|E_y|^2+|E_z|^2$",
           "vectorial ŷ (total)")]
for ax, A, title, name in panels:
    show_image(ax, A, title)
    draw_cuts(ax)
    annotate_C(ax, name)
fig.suptitle(rf"Proyección de inmersión LuAG$\to$agua: $\lambda=193$ nm, NA$={NA:.2f}$, reducción {REDUCTION:.0f}$\times$, Köhler $\sigma={SIGMA}$, iluminación lineal")
fig.savefig(out_dir / "07_linear_images.png", dpi=200)
plt.show()

# %% [markdown]
# ## La evidencia en detalle: acercamientos con la máscara como referencia
#
# Cada fila acerca uno de los dos grupos de trazas; la primera columna es el
# recorte de la máscara — cómo debería verse la imagen idealmente.

# %%
zoom_V = (2.2, 9.6, 0.2, 5.6)
zoom_H = (-6.0, -0.4, -12.6, -7.6)
cols = [("escalar", I_esc, "escalar"),
        ("vectorial x̂ (total)", I_lin["vectorial x̂ (total)"], r"vectorial $\hat{x}$ (total)"),
        ("vectorial ŷ (total)", I_lin["vectorial ŷ (total)"], r"vectorial $\hat{y}$ (total)")]
fig, axes = plt.subplots(2, 4, figsize=(17.6, 9.6), constrained_layout=True)
for i, (zoom, cut, tag) in enumerate([(zoom_V, cut_V, "trazas verticales"),
                                      (zoom_H, cut_H, "trazas horizontales")]):
    show_mask(axes[i, 0], zoom=zoom, title=f"máscara — {tag}")
    axes[i, 0].text(0.02, 0.99, "$C$ = 1.00", transform=axes[i, 0].transAxes,
                    va="top", color="w", fontsize=11)
    for j, (name, A, title) in enumerate(cols, start=1):
        show_image(axes[i, j], A, title, zoom=zoom, cmap="hot")
        Cval = C[name][0] if i == 0 else C[name][1]
        lab = "$C_V$" if i == 0 else "$C_H$"
        axes[i, j].text(0.02, 0.99, f"{lab} = {Cval:.2f}",
                        transform=axes[i, j].transAxes, va="top",
                        color="w", fontsize=11)
    for j in range(4):
        if cut["horizontal"]:
            axes[i, j].plot([cut["t_min"] / lam, cut["t_max"] / lam],
                            [cut["axis_value"] / lam] * 2, "c-", lw=1.0)
        else:
            axes[i, j].plot([cut["axis_value"] / lam] * 2,
                            [cut["t_min"] / lam, cut["t_max"] / lam], "c-", lw=1.0)
fig.savefig(out_dir / "07_zooms.png", dpi=200)
plt.show()

# %% [markdown]
# ### Desglose por componentes en los acercamientos
#
# Los mismos acercamientos, canal por canal, para la iluminación $\hat{x}$.
# Todos los paneles comparten la **misma escala de intensidad** (la de
# $I_{\rm tot}$); solo $|E_y|^2$ va amplificado — su peso real es
# $f_{\rm cross} \sim 10^{-3}$ — con el factor indicado en el título.

# %%
I_tot_x = I_lin["vectorial x̂ (total)"]
v_ref = np.percentile(I_tot_x, 99.6)
I_ey_x = np.abs(img_vx.y) ** 2
amp_y = float(f"{v_ref / np.percentile(I_ey_x, 99.6):.1g}")
chan_cols = [
    (np.abs(img_vx.x) ** 2, 1.0, r"$|E_x|^2$ (co-polarizado)"),
    (I_ey_x, amp_y, rf"$|E_y|^2$ (cruzado, $\times{amp_y:g}$)"),
    (np.abs(Ez_x) ** 2, 1.0, r"$|E_z|^2$ (longitudinal)"),
    (I_tot_x, 1.0, r"$I_{\rm tot}$ (suma)"),
]


def show_channel(ax, A, title, zoom, gain=1.0):
    B = A * gain
    ext = [xr[0] / lam, xr[-1] / lam, yr[0] / lam, yr[-1] / lam]
    ax.imshow(np.clip(B / v_ref, 0, 1) ** 0.85, origin="lower", extent=ext,
              cmap="inferno")
    ax.set_xlim(zoom[0], zoom[1])
    ax.set_ylim(zoom[2], zoom[3])
    ax.set(title=title, xlabel=r"$x/\lambda$", ylabel=r"$y/\lambda$")


fig, axes = plt.subplots(2, 5, figsize=(21.0, 9.4), constrained_layout=True)
for i, (zoom, tag) in enumerate([(zoom_V, "trazas verticales"),
                                 (zoom_H, "trazas horizontales")]):
    show_mask(axes[i, 0], zoom=zoom, title=f"máscara — {tag}")
    for j, (A, gain, title) in enumerate(chan_cols, start=1):
        show_channel(axes[i, j], A, title, zoom, gain=gain)
fig.suptitle(r"iluminación $\hat{x}$: canales del campo en los acercamientos"
             " (escala común)", fontsize=13)
fig.savefig(out_dir / "07_zoom_channels_x.png", dpi=200)
plt.show()

# %% [markdown]
# El canal longitudinal **se enciende sobre las trazas verticales** (fila
# superior): ahí es donde $I_{\rm tot}$ pierde el valle que $|E_x|^2$ solo
# todavía conserva en parte. Sobre las trazas horizontales (caso TE, fila
# inferior) $|E_z|^2$ es débil y la suma respeta la máscara. $|E_y|^2$ es
# anecdótico en ambas: la diatenuación de Fresnel no participa aquí.
#
# ## Perfiles sobre los cortes: iluminación $\hat{x}$
#
# Solo el caso $\hat{x}$ (el $\hat{y}$ es su imagen especular). Las bandas
# grises marcan las trazas de la máscara — el perfil ideal.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6), constrained_layout=True)
curves = [("escalar", I_esc, "k", "--"),
          ("vectorial x̂ (transversal)", I_lin["vectorial x̂ (transversal)"],
           "tab:orange", ":"),
          ("vectorial x̂ (total)", I_lin["vectorial x̂ (total)"], "tab:orange", "-")]
for ax, cut, centers, tag, ci in [(axes[0], cut_V, centers_V, "corte V", 0),
                                  (axes[1], cut_H, centers_H, "corte H", 1)]:
    for name, A, color, ls in curves:
        t, vals = image_cut(A, cut)
        ax.plot(t / lam, vals / vals.max(), ls, color=color, label=name)
    for c in centers:
        ax.axvspan((c - 7 * dx) / lam, (c + 7 * dx) / lam, color="0.90", zorder=0)
    Ctxt = "\n".join(f"C = {C[name][ci]:.2f}  ({name})" for name, *_ in curves)
    ax.text(0.02, 0.98, Ctxt, transform=ax.transAxes, va="top", fontsize=8)
    ax.set(title=tag, ylabel=r"$I/I_{\max}$",
           xlabel=(r"$x/\lambda$" if cut["horizontal"] else r"$y/\lambda$"))
    ax.set_ylim(0, 1.28)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, 1.09))
fig.savefig(out_dir / "07_profiles_x.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# En el corte V (modulación $\parallel \hat{x}$) el vectorial pierde el valle
# en dos pasos: la parte transversal (proyección TM entre órdenes) y el
# relleno por $|E_z|^2$. En el corte H (modulación $\perp \hat{x}$, caso TE)
# las tres curvas siguen de cerca al escalar: el contraste conserva la mayor
# parte ($0.98 \to 0.85$, contra $0.91 \to 0.67$ en V).
#
# ## ¿A dónde se fue el contraste? Canales de la iluminación $\hat{x}$
#
# Primero las dos intensidades que se comparan (escalar y vectorial total),
# luego su diferencia, y los tres canales del campo vectorial.

# %%
panels = [
    (I_esc, r"$I_{\rm esc}$ (escalar, $t_p = t_s = 1$)", "gray"),
    (I_tot_x, r"$I_{\rm tot}$ (vectorial $\hat{x}$)", "gray"),
    (np.abs(I_tot_x / I_tot_x.mean() - I_esc / I_esc.mean()),
     r"$|I_{\rm tot} - I_{\rm esc}|$ (norm. media)", "inferno"),
    (np.abs(img_vx.x) ** 2, r"$|E_x|^2$ (co-polarizado)", "inferno"),
    (np.abs(img_vx.y) ** 2,
     rf"$|E_y|^2$ (cruzado, $f_{{\rm cross}}$ = {f_cross:.4f})", "inferno"),
    (np.abs(Ez_x) ** 2, rf"$|E_z|^2$ (longitudinal, $f_z$ = {f_z:.2f})", "inferno"),
]
fig, axes = plt.subplots(2, 3, figsize=(17.0, 11.6), constrained_layout=True)
for ax, (A, title, cmap) in zip(axes.ravel(), panels):
    show_image(ax, A, title, cmap=cmap)
fig.savefig(out_dir / "07_channels_x.png", dpi=200)
plt.show()

# %% [markdown]
# $|E_z|^2$ no es un velo uniforme: se concentra sobre las trazas
# **verticales** (las de modulación $\parallel \hat{x}$), exactamente donde el
# contraste se pierde — es el mapa del mecanismo.
#
# ## Mapas de polarización en el plano imagen
#
# Elipses de polarización **transversales** ($E_x$, $E_y$) sobre el fondo de
# intensidad transversal, en los mismos acercamientos: campo incidente (el
# estado ideal) contra la imagen vectorial. La lección es doble. Primero, la
# elipse transversal permanece esencialmente lineal según $\hat{x}$: con
# $f_{\rm cross} \sim 10^{-3}$ la diatenuación de Fresnel apenas rota o abre
# el estado — al contrario que en los sistemas de los cuadernos 01–04, donde
# era el canal protagonista. Segundo, y por lo mismo: **la aberración de este
# régimen no se ve en el mapa transversal** — vive en $E_z$, que no cabe en
# una elipse transversal (columna $|E_z|^2$ del desglose anterior).

# %%
x_m = xr
y_m = yr


def image_field_window(fld, zoom):
    """Image-plane field rotated to mask coordinates, cropped to a window (λ)."""
    sx = (x_m / lam >= zoom[0]) & (x_m / lam <= zoom[1])
    sy = (y_m / lam >= zoom[2]) & (y_m / lam <= zoom[3])
    Ex = fld.x[np.ix_(sy, sx)]
    Ey = fld.y[np.ix_(sy, sx)]
    s = np.sqrt(np.max(np.abs(Ex) ** 2 + np.abs(Ey) ** 2))  # colorbar en [0, 1]
    return FieldCartesian(Ex / s, Ey / s,
                          grid=Grid.from_axes(x_m[sx] / lam, y_m[sy] / lam),
                          symmetric=False)


def incident_field_window(Ex0, Ey0, zoom):
    sx = (x_ax / lam >= zoom[0]) & (x_ax / lam <= zoom[1])
    sy = (x_ax / lam >= zoom[2]) & (x_ax / lam <= zoom[3])
    return FieldCartesian(np.asarray(Ex0, dtype=complex)[np.ix_(sy, sx)],
                          np.asarray(Ey0, dtype=complex)[np.ix_(sy, sx)],
                          grid=Grid.from_axes(x_ax[sx] / lam, x_ax[sy] / lam),
                          symmetric=False)


pol_kwargs = dict(sampling="cartesian", target_ellipses=23,
                  scale_by_intensity=True, min_ellipse_scale=0.5)

fig, axes = plt.subplots(2, 2, figsize=(16.6, 12.4), constrained_layout=True)
for i, (zoom, tag) in enumerate([(zoom_V, "trazas verticales"),
                                 (zoom_H, "trazas horizontales")]):
    plot_field_polarization(incident_field_window(T, Z0, zoom),
                            ax=axes[i, 0], **pol_kwargs)
    axes[i, 0].set_title(rf"incidente $T\hat{{x}}$ — {tag}")
    plot_field_polarization(image_field_window(img_vx, zoom),
                            ax=axes[i, 1], **pol_kwargs)
    axes[i, 1].set_title(f"imagen vectorial (transversal) — {tag}")
    for ax in axes[i]:
        ax.set(xlabel=r"$x/\lambda$", ylabel=r"$y/\lambda$")
fig.savefig(out_dir / "07_polarization_x.png", dpi=200)
plt.show()

# %% [markdown]
# ## Iluminación circular
#
# Campo incidente $\mathbf{E}_0 = T(\hat{x} + i\hat{y})/\sqrt{2}$, intensidad
# total con $|E_z|^2$. Como en los cuadernos 03 y 06, la incidencia circular
# **isotropiza el costo vectorial, no lo elimina**.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.0), constrained_layout=True)
show_image(axes[0], I_esc, "escalar ($t_p = t_s = 1$)")
draw_cuts(axes[0])
annotate_C(axes[0], "escalar")
show_image(axes[1], I_cir,
           r"vectorial, $\mathbf{E}_0 = T(\hat{x}+i\hat{y})/\sqrt{2}$ — total")
draw_cuts(axes[1])
annotate_C(axes[1], "circular (total)")
fig.savefig(out_dir / "07_circular_images.png", dpi=200)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6), constrained_layout=True)
for ax, cut, centers, tag, ci in [(axes[0], cut_V, centers_V, "corte V", 0),
                                  (axes[1], cut_H, centers_H, "corte H", 1)]:
    for name, A, color, ls in [("escalar", I_esc, "k", "--"),
                               ("circular (total)", I_cir, "tab:purple", "-")]:
        t, vals = image_cut(A, cut)
        ax.plot(t / lam, vals / vals.max(), ls, color=color, label=name)
    for c in centers:
        ax.axvspan((c - 7 * dx) / lam, (c + 7 * dx) / lam, color="0.90", zorder=0)
    Ctxt = (f"C = {C['escalar'][ci]:.2f}  (escalar)\n"
            f"C = {C['circular (total)'][ci]:.2f}  (circular total)")
    ax.text(0.02, 0.98, Ctxt, transform=ax.transAxes, va="top", fontsize=8)
    ax.set(title=tag, ylabel=r"$I/I_{\max}$",
           xlabel=(r"$x/\lambda$" if cut["horizontal"] else r"$y/\lambda$"))
    ax.set_ylim(0, 1.28)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, 1.09))
fig.savefig(out_dir / "07_profiles_circ.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# El mapa de polarización del caso circular, en el acercamiento de las trazas
# verticales: el estado transversal de la imagen sigue siendo esencialmente
# circular (misma razón que en el caso lineal: $f_{\rm cross}$ minúsculo). El
# costo vectorial — ahora repartido entre ambas orientaciones — vuelve a estar
# en $E_z$, invisible para la elipse transversal.

# %%
fig, axes = plt.subplots(1, 2, figsize=(16.6, 6.6), constrained_layout=True)
plot_field_polarization(
    incident_field_window(T / np.sqrt(2.0), 1j * T / np.sqrt(2.0), zoom_V),
    ax=axes[0], **pol_kwargs)
axes[0].set_title(r"incidente $T(\hat{x}+i\hat{y})/\sqrt{2}$ — trazas verticales")
plot_field_polarization(image_field_window(img_ci, zoom_V),
                        ax=axes[1], **pol_kwargs)
axes[1].set_title("imagen vectorial (transversal) — trazas verticales")
for ax in axes:
    ax.set(xlabel=r"$x/\lambda$", ylabel=r"$y/\lambda$")
fig.savefig(out_dir / "07_polarization_circ.png", dpi=200)
plt.show()

# %% [markdown]
# ## Conclusión: la aberración vectorial en su forma aplicada
#
# El sistema no tiene **ninguna** aberración geométrica: dioptrios
# estigmáticos, conjugación exacta, aumento unitario. El modelo escalar
# ($t_p = t_s = 1$) predice una fidelidad de imagen esencialmente
# **independiente de la orientación** de las trazas ($C_V - C_H = -0.07$, la
# pequeña diferencia geométrica entre los dos grupos). El campo vectorial no
# cumple esa promesa:
#
# - Con iluminación $\hat{x}$, las trazas cuya modulación es **paralela** a la
#   polarización (corte V) pierden contraste ($0.91 \to 0.67$): las órdenes
#   difractadas se separan en el plano $x$–$z$, sus campos dejan de ser
#   paralelos (proyección TM $\propto \cos\theta_{\rm mutuo} \approx 0.59$
#   para el paso fino) y $E_z$ — que aquí carga $f_z \approx 0.5$ de la
#   energía — rellena los valles. Las trazas con modulación **perpendicular**
#   (corte H) son TE y conservan la mayor parte del contraste
#   ($0.98 \to 0.85$): la firma $C_V - C_H$ pasa de $-0.07$ (escalar) a
#   $-0.17$.
# - Con iluminación $\hat{y}$ el papel de los cortes se intercambia
#   ($C_V - C_H = +0.14$), y la iluminación **circular** reparte la pérdida
#   entre ambas orientaciones ($-0.01$): la isotropiza, no la elimina —
#   exactamente lo que mostró la incidencia circular en los cuadernos 03 y 06.
# - La jerarquía de mecanismos se invierte respecto de los cuadernos 04–06: la
#   diatenuación de Fresnel es aquí secundaria ($f_{\rm cross} \sim 0.1\%$) y
#   la **transversalidad** domina ($f_z \approx 0.5$). Los dos canales
#   vectoriales son independientes, como estableció el cuaderno 05: éste es el
#   extremo donde manda $E_z$.
#
# Es la conclusión central de la investigación en su versión más tangible: un
# diseño validado con teoría escalar imprimiría este circuito con trazas
# nítidas en una orientación y degradadas en la ortogonal — una **aberración
# de origen netamente vectorial**, invisible para el modelo escalar, presente
# en un sistema libre de aberración geométrica. Es el mismo fenómeno que
# obliga a la litografía hiper-NA a controlar la polarización de la
# iluminación [Flagello *et al.* 1996; Ruoff & Totzeck 2009].
