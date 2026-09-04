# Investigación: tamaño y forma del lóbulo central — escalar vs. vectorial

¿Cuánto difiere el punto focal del modelo vectorial respecto del escalar, y
qué controla su deformación? Para polarización lineal homogénea a ángulo α el
"radio del punto" no basta: el lóbulo tiene **dos radios principales**
(contraído a lo largo de α, expandido en la ortogonal).

## Reducción analítica que organiza todo el estudio

Para pupila con polarización lineal homogénea (ángulo α) y amplitud radial
`P(r)`, la reconstrucción cartesiana de `vecdiff` da exactamente

```
I(ρ,φ) = π² [ |H0|² + |H2|² − 2 Re(H0 H2*) cos 2(φ−α) ]
H0 = 𝓗₀[(tp+ts)P] ,  H2 = 𝓗₂[(tp−ts)P]
```

donde `tp`, `ts` son los **pesos efectivos** de canal: los coeficientes de
Fresnel por el factor geométrico `A(Q) = |z0|ℓi/(|zi|ℓ0)` (con la proyección
meridional en el canal p) y por el jacobiano `1/cos αᵢ` del mapeo de pupila.
Las transformadas 𝓗ₘ integran sobre `u = zi sin αᵢ`, no sobre el radio de
pupila `r` — ver `vecdiff.transfer` y `vecdiff.pupil_mapping`.

- El patrón **rota rígidamente** con α (verificado a precisión de máquina):
  basta estudiar α=0.
- Cortes principales: `I∥ = π²|H0−H2|²` (a lo largo de α) y `I⊥ = π²|H0+H2|²`.
- Referencias escalares: *apodizada* (`t−=0`, perfil `|H0|²`, gratis) e
  *ideal* (`tp=ts=1`, Airy si `P=1`).
- Incidencia circular: `I ∝ |H0|²+|H2|²`, simétrica (= promedio azimutal del
  caso lineal).

## Cuadernos

| Cuaderno | Contenido | Hallazgos principales |
| --- | --- | --- |
| `01_baseline_spot_metrics` | Métricas (HWHM, primer mínimo, elipticidad, f_cross), validación vs. Airy y vs. la tubería completa, rotación rígida, incidencia circular, barrido de apertura. | `R∥ < R_sca < R⊥` (contracción ∥ / expansión ⊥); la apodización de Fresnel `t₊(r)` ensancha el punto más de lo que la mezcla vectorial lo deforma. |
| `02_cross_maximization` | Qué maximiza el canal `H2`: peso de Fresnel `t−(r)`, solapamiento con `J2(qr)`, familias de borde `(r/a)^p` y anulares, apertura × índice, techo del anillo rasante `ρ²/(1+ρ²)`, transmisión. | La cruzada se genera en la **zona externa** de la pupila; jerarquía de palancas: apertura ≫ índice (fija el techo) > forma de pupila (paga transmisión). |
| `03_deformation_vs_cross` | Atlas apertura × índice, tres familias de pupilas, predictor de primer orden, incidencia circular. | La deformación **no** la gobierna `f_cross` (2º orden, global) sino el solapamiento local `Re(H0H2*)` en el radio de mitad de altura: el predictor `A_pred = 4B(R_sca)/(|I'_sca| R_sca)` colapsa todas las familias a ~1%. |
| `04_polarization_aberrations` | Aberraciones de polarización à la Chipman (McGuire & Chipman, JOSA A 7, 1614): pupila de Jones, descomposición de Pauli, glifos de diatenuación, coeficientes de segundo orden, conexión con el foco. | El dioptrio es un diatenuador radial puro; `f_cross = ⟨(1−√(1−D²))/2⟩` exacto (6e-13) en todo el atlas, con `D_rms²/4` como orden dominante (≲3.5% en el peor caso); anisotropía `A ≈ κ·D₂a²/2` con `κ = 2J₃(u_h)/(u_h J₂(u_h)) ≈ 0.353`, ahora a −0.5% de la medida. La expansión paraxial de los pesos **efectivos** requiere `A(Q)` y el jacobiano además de los coeficientes de Fresnel, y da al canal cruzado un término puramente geométrico. |
| `05_longitudinal` | La componente longitudinal `\|Ez\|²` en forma radial cerrada: `Ez = −2πi·𝓗₁[w_z tp P]·cos(φ−α)`, `w_z = tan αᵢ` (exacto; la forma `(r/zi)/√(1−(r/zi)²)` que había aquí es su aproximación de superficie delgada); validación contra `vecdiff.longitudinal`; cortes, energías `f_z`, mapas anotados totales, barrido de apertura. | `I⊥` no cambia; `I∥` gana `4\|Hz\|²`. Predictor extendido `A_tot ≈ 4(B−Z)/(\|I'\|R_sca)`: la elongación se invierte (régimen Richards–Wolf) donde `Z=B`. En vidrio `f_z ≈ 15·f_cross` y los dos mecanismos se compensan en todo el barrido (`\|A_tot\|≲0.004`: lóbulo total redondo); en el dioptrio rápido ni=2.4 domina Fresnel (`e_tot=1.098`). |
| `06_two_point_resolution` | Resolución de dos puntos, escalar vs. vectorial: sobre el eje de separación cada fuente aporta el corte principal (`∥`: `π[H0−H2]` con `Ez` impar que se cancela en el punto medio coherente en fase; `⊥`: `π[H0+H2]` con `Ez≡0`; circular = `½(I∥_tot+I⊥)` exacto); límites de Sparrow coherentes e incoherentes con y sin `\|Ez\|²`; mapas 2D vía `field_components`. | El vectorial es **sistemáticamente menos optimista** en `⊥` (hasta +8.7% incoherente con pupila de borde) y en circular; en `∥` hay una ganancia direccional modesta (hasta ~7%, erosionada por `4\|Hz\|²` en incoherente). Caso dramático: la pupila de borde `(r/a)^8` superresolvente escalar **fusiona** lo que el escalar resuelve — aberración de resolución de origen netamente vectorial en un sistema sin aberración geométrica. |
| `07_lithography_pattern` | Objeto extendido no simétrico: patrón de circuito (`../circuit_pattern.png`, **rectificado**: cada traza dibujada a mano se reemplaza por un rectángulo recto de ancho de diseño 14 px) impreso por un sistema de proyección de inmersión (pupila LuAG→agua, NA=0.88, reducción 4×) con iluminación Köhler σ=0.2 (método de Abbe) y el paso fino en 1.02× el corte coherente; tres modos de pupila: escalar puro (`t_p=t_s=1`, sin apodizar), escalar físico (canal s: `A·t_s·jac`, la solución exacta del problema escalar de contorno, verificada a 1.6e-8 contra Helmholtz–Kirchhoff) y vectorial `x̂`/`ŷ`/circular con `Ez` en la suma de Abbe; contraste de trazas sobre cortes V y H. | **El escalar resuelve el circuito y el vectorial no lo imprime**: escalar `C_V=0.95, C_H=0.81` contra vectorial total ~0.43/0.39 — bajo un umbral de imprimibilidad de 0.5. El desglose por modos muestra que en esta pupila domina la **apodización física de amplitud** (el escalar físico ya cae a ~0.45, casi igual al vectorial) y la polarización aporta la firma direccional encima: a igual σ, el canal TM pierde ~0.06 más que el TE (`C_H`: 0.397 con `x̂` → 0.338 con `ŷ`), y `f_z = 0.26` de la energía va al canal longitudinal. Es el efecto TE/TM de la litografía hiper-NA (Flagello et al., JOSA A 13, 53; Ruoff & Totzeck 2009). |

`spot_tools.py` contiene la maquinaria compartida (transformadas de Hankel
vectorizadas, radio rasante, perfiles, métricas, energías por Parseval,
mapas anotados con radios medidos — `plot_spot_with_radii`, con `total=True`
para incluir `|Ez|²` — el canal longitudinal `Hz`/`w_z`/`f_z`, los campos
complejos 2D `field_components` (Ex, Ey, Ez para incidencia lineal o
circular, validados a precisión de máquina contra la reconstrucción del
paquete), y métricas de diatenuación de Chipman).

Los radios medidos (contorno HWHM, radios principales, referencia escalar)
van **anotados sobre los mapas de intensidad**: `01_annotated_spots.png` para
un dioptrio rápido de alto índice (`ni=2.4, z0=−2, zi=6`, α_obj≈57°, donde la
deformación es evidente: e≈1.14; el vidrio la limita a e≲1.08) y
`02_pattern_effect.png` para las pupilas que maximizan la cruzada. La ventana
de estas figuras se recorta al lóbulo central (`display_half_size`): a lo
sumo tres máximos secundarios visibles.

## Convenciones

- Radios focales en unidades de λ: `s = ρ/λ = q·q_λ`, con `q_λ = zi/(2π ni)`.
- Dioptrio base: `n0=1, ni=1.5, z0=−10 mm, zi=6 mm, λ=532 nm` (el de
  `examples/aperture_scalar_vs_vectorial.py`); alto índice: `ni=2.4`.
- Radio rasante `r_graze`: radio **transversal sobre la superficie**, que es
  la coordenada de apertura. No es el parámetro `ρ` del óvalo, que en el mismo
  punto vale más (3.758 contra 2.397 en el sistema base).
- Las energías de canal (`f_cross`) se calculan **en el espacio de pupila vía
  Parseval** (`∫|𝓗ₘ[f]|² q dq = ∫|f|² u du`, con la medida del mapeo): la
  integración numérica en q con grillas alcanzables es oscilatoria y no
  converge (véase nota abajo).
- La `transmisión` es un cociente de flujo genuino entre las dos esferas de
  referencia, no la integral de `|E|²` en el plano focal: los rayos cruzan ese
  plano oblicuamente, así que esa integral es una medida de forma bien
  definida pero no la potencia que lo atraviesa.

## Cómo regenerar

Los `.ipynb` se generan y ejecutan desde los `.py` (formato jupytext percent),
desde esta carpeta:

```bash
uv run jupytext --to ipynb --execute 01_baseline_spot_metrics.py
```

Las figuras quedan en `output/`.

## Nota numérica

`f_cross` integrado en q con la grilla estirada de los ejemplos no converge
(el integrando `|H|²q` oscila con período `2π/a`): para el caso uniforme de
alto índice da 0.021–0.036 según la grilla, cuando el valor exacto (Parseval)
es 0.0169. Esto afecta también a los valores impresos por
`examples/maximize_cross_polarization.py`.

## Hoja de ruta

- ~~Incluir el campo longitudinal `|Ez|²` en las métricas~~ — hecho
  (cuaderno 05): invierte la elongación donde `Z=B` en `R_sca`.
- ~~Propagar las métricas al plano imagen del sistema de dos dioptrios~~ —
  hecho (cuaderno 07): relevo 1:1, contraste por orientación en el plano
  imagen.
- Polarizaciones no homogéneas (radial/azimutal, base `polar`): el canal `J2`
  se reemplaza por `J1`.
- Dependencia axial (fuera de foco) de los radios principales.
