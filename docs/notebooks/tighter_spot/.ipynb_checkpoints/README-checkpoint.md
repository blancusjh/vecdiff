# Investigación: el spot más pequeño — replicación de Quabis et al. (2000)

¿Concuerda el canal longitudinal `E_z` que calcula `vecdiff` con la
literatura de enfoque fuerte? Referencia canónica: Quabis, Dorn, Eberler,
Glöckl y Leuchs, "Focusing light to a tighter spot", Opt. Commun. **179**,
1–7 (2000), y su confirmación experimental Dorn, Quabis y Leuchs, PRL
**91**, 233901 (2003): a NA alta, la dona radialmente polarizada enfoca a
un área menor que la polarización lineal porque `E_z` (canal `J₀`, peso
`tanθ`) domina y su lóbulo central es el más angosto.

## Cuadernos

| Cuaderno | Contenido | Hallazgos principales |
| --- | --- | --- |
| `01_quabis2000_replication` | Réplica de la Tabla 1 (áreas de spot a mitad de altura, 7 iluminaciones × NA ∈ {0.7, 0.8, 0.9, 1.0}), réplicas fieles de las Figs. 5 y 6 (perfiles + mapas de contorno 2D en ±2λ) y spots anotados a NA=0.9 —configuración de Dorn 2003— con contorno de media altura, radios principales `R_∥/R_⊥` y áreas medidas vs. publicadas. Integrales de Debye–Wolf aplanáticas (`√cosθ`, receta Richards–Wolf), cuadratura directa en θ. | La tabla se reproduce a 1–4 % en casi todas las entradas (Dorn NA=0.9: 0.175 vs 0.166 λ²; fila «solo E_z»: 0.375/0.284/0.221/0.169 vs 0.373/0.275/0.212/0.160). A NA=0.9 el cociente `A(dona+anular)/A(lineal)` da 0.56 (artículo: 0.54). Las dos desviaciones grandes (lineal anular NA=1, dona total NA=0.7) son entradas cuyo contorno de media altura cae en zonas de pendiente casi nula: el área es allí hipersensible a la discretización. El «spot más pequeño» radial es un efecto de alta apertura: a NA=0.7 el anillo transversal `J₁` domina (2.5 λ² ≫ 0.46 λ² lineal). |
| `02_quabis2000_vecdiff` | La misma réplica con el **motor de la tesis**: la sustitución `u = sinθ` convierte las integrales de Debye–Wolf en transformadas de Hankel evaluadas con `vecdiff.hankel.HankelTransform` (pupila generalizada aplanática), y `E_z` se reconstruye desde las componentes transversales con `vecdiff.longitudinal.generate_Ez_cartesian` (espectro angular). Figuras nuevas: mapas cmap-hot por componentes a NA=0.9 (copolar/cruzada/E_z lineal; E_⊥/E_z/total para dona y dona+anular) y superposición de contornos de mitad de altura (escalar, lineal, dona, dona+anular). | La Tabla 1 vía vecdiff es **idéntica** a la del cuaderno 01. El `E_z` de vecdiff coincide con el canal de Richards–Wolf: 2.8e-3 (lineal), 1.7e-3 (dona), 1.4e-2 (dona+anular; sus colas `ρ^(-1/2)` exigen apodizar la ventana FFT con un `cos²` entre 12λ y 15.5λ). Orden de spots a NA=0.9: dona+anular (0.175 λ²) < **escalar** (0.240 λ²) < lineal (0.314 λ²) < dona (0.670 λ²): el spot radial bate incluso a la predicción escalar — `E_z` es un canal que la teoría escalar no contiene. |

## Conexión con la tesis

El cociente de pesos radial `sin²θ/(sinθcosθ) = tanθ` de las integrales
aplanáticas es exactamente el peso `w_z` del canal longitudinal del
formalismo del dioptrio estigmático; la diferencia entre ambos modelos es la
apodización (`√cosθ` aplanática vs. `t_p` de Fresnel). El dioptrio de vidrio
de los ejemplos (`sinθ_max ≈ 0.41`, NA_i ≈ 0.62) queda en el régimen «anillo
dominante» (`I_z(0) ≈ 0.3·max I_⊥`): el cruce al régimen de Quabis
(`E_z` dominante) ocurre entre `sinθ_max = 0.7` y `0.85`.

## Hoja de ruta

- Cuaderno 03: el mismo experimento numérico sobre el dioptrio estigmático
  (apodización de Fresnel `t_p`, propagador de dioptrio de `vecdiff`),
  barriendo la apertura hasta el cruce al régimen de Quabis; dona radial
  vs. iluminación uniforme.

## Cómo regenerar

```bash
../../../.venv/bin/jupytext --to ipynb --execute 01_quabis2000_replication.py
../../../.venv/bin/jupytext --to ipynb --execute 02_quabis2000_vecdiff.py
```

Las figuras quedan en `output/`. El cuaderno 02 tarda ~4 min: el bucle
sobre `q` de `HankelTransform.transform_array` es Python puro (~2.4 s por
canal de 1501 muestras).
