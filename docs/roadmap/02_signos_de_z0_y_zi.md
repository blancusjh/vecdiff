# Signos de z0 y zi: objetos e imágenes virtuales, y espejos

**Objetivo:** que `vecdiff` acepte `z0` y `zi` de cualquier signo —no solo la
convención `z0 < 0 < zi`— cubriendo objetos e imágenes *virtuales* (refracción)
y *espejos* (reflexión), con los resultados validados contra el solver de
Maxwell (Franz / Stratton–Chu).

> Nota de sesión: este documento recoge **hallazgos**, no cambios de código. La
> maquinaria física no se tocó (la exploración de signos que se había hecho se
> revirtió a propósito). Sirve de punto de partida para una línea de trabajo
> dedicada.

## El mapa de casos

Con el vértice en `z = 0`, luz viajando hacia `+z`, medio `n0` en el lado
objeto y `ni` en el lado imagen, los cuatro signos de `(z0, zi)` son estados
físicos distintos:

| `z0` | `zi` | objeto | imagen | condición del óvalo |
| --- | --- | --- | --- | --- |
| `< 0` | `> 0` | real | real | `n0 l0 + ni li = const` (suma) |
| `< 0` | `< 0` | real | **virtual** | `n0 l0 − ni li = const` (resta) |
| `> 0` | `> 0` | **virtual** | real | `n0 l0 − ni li = const` (resta) |
| `> 0` | `< 0` | **virtual** | **virtual** | `n0 l0 + ni li = const` (suma) |

Lo que decide refracción vs. reflexión es el **índice**, no el signo de las
distancias:

- `|n0| ≠ |ni|` → **refracción**. El signo de `z0, zi` fija real/virtual.
- **espejo** → convención de índice con signo `ni = −n0`. Entonces
  `n0 l0 + ni li` se vuelve `n0 (l0 − li)`, es decir la ley de reflexión:
  `z0 < 0, zi < 0` da `l0 + li = const` (espejo elíptico) y `z0 < 0 < zi` da
  `l0 − li = const` (espejo hiperbólico).
- `n0 = +ni` exacto es singular (no hay superficie que refracte entre medios
  idénticos): `GOTS_params` (`vecdiff/GOTS_parameters.py`) divide por
  `(ni − no)`. Es el caso degenerado, correctamente excluido.

## El hallazgo que lo unifica: distancias con signo

Si `l0` y `li` se definen como **distancias con signo** —positivas cuando la
medición va de un punto a otro en el sentido de `z` creciente, negativas cuando
`z` decrece—

```
l0 = sign(z_Q − z_A) · |A→Q|          (medida A → Q)
li = sign(z_A' − z_Q) · |Q→A'|        (medida Q → A')
```

entonces **una sola fórmula** vale para todos los casos, sin ramas suma/resta:

```
n0 · l0 + ni · li = const = −n0·z0 + ni·zi
```

Verificado numéricamente sobre el óvalo cerrado actual (`CartesianSurface.z`),
desviación respecto de `−n0·z0 + ni·zi`:

| caso | signos | desviación |
| --- | --- | --- |
| real → real | `z0<0<zi` | `3.6e-15` |
| real → virtual | `z0<0, zi<0` | `1.8e-15` |
| virtual → real | `z0>0, zi>0` | `1.8e-15` |
| virtual → virtual | `z0>0>zi` | `3.6e-15` |
| espejo elíptico | `ni=−n0, z0<0, zi<0` | `1.8e-15` |
| espejo hiperbólico | `ni=−n0, z0<0<zi` | `3.6e-15` |

**Conclusión: la superficie ya es correcta para todos los casos.** El óvalo
cerrado (`GOTS_params` + `CartesianSurface.z`) satisface la condición
estigmática con signo a precisión de máquina en los seis. Lo que falta está
**aguas abajo**: la geometría de rayos, el operador de transferencia y los
propagadores suponen `z0 < 0 < zi` (objeto y imagen reales, rayo transmitido
hacia `+z`).

## Límites actuales

**Geometría de rayos** (`vecdiff/geometry.py`). Marca inválidos los casos de
mismo signo (objeto/imagen virtual) en el vértice:

- `orientation = 1.0 if z0 > 0.0 else -1.0` (`geometry.py:267`) orienta la
  normal solo por el signo de `z0`.
- `refracting = (cos_t0 > 0.0) & (cos_ti > 0.0)` (`geometry.py:289`) usa los
  cosenos de incidencia con la dirección de rayo `A→Q`/`Q→A'`, que para objeto
  o imagen virtual apunta al revés que la luz física; el rayo axial ya falla y
  todo el radio queda `valid = False`.
- `grazing_radius` lanza `ValueError` con el mensaje «the physical convention
  is z0 < 0 < zi» (`geometry.py:348-351`). El mensaje es incorrecto: el
  mismo-signo no es «otro óvalo», es objeto/imagen virtual.
- El factor geométrico `A = |z0| li / (|zi| l0)` y los cosenos `cos_a0`,
  `cos_ai` se construyen con `l0, li` sin signo; con distancias con signo hay
  que rederivarlos.

**Mapeo de pupila** (`vecdiff/pupil_mapping.py`). El jacobiano de Debye
`1/cos(alpha_i)` y la coordenada tangente se guardan con `cos_ai > 0.0`
(`pupil_mapping.py:59`, `pupil_mapping.py:72`). Para imagen virtual (`zi < 0`)
el rayo transmitido va hacia `−z`, `cos_ai < 0`, y la guarda **anula todo el
integrando** → campo focal idénticamente cero. El ángulo sólido depende solo de
`|cos(alpha_i)|`.

**Propagación** (`vecdiff/propagation.py`). La escala de la malla focal usa
`diopter.zi` con signo (`propagation.py:403`): `zi < 0` refleja la malla en vez
de dejar una coordenada radial limpia (debería ser `|zi|`, como el resto).

**Transferencia paraxial** (`vecdiff/transfer.py`). `A_slope` usa
`(1/|zi| + 1/|z0|)` (`transfer.py:168`). Bajo el espejo `z → −z`, `O` cambia de
signo pero `|z0|, |zi|` no, así que el término no es invariante; la forma con
signo `(1/zi − 1/z0)` sí lo es y coincide con la actual para `z0 < 0 < zi`.

**Relay de dos superficies** (`vecdiff/system.py`). `np.maximum(g1.cos_ai, ...)`
(`system.py:124`) recorta a `1e-15` cuando `cos_ai < 0` en vez de tomar
`|cos_ai|`.

**Solver de referencia** (`vecdiff/reference/`). Comparte `ray_geometry`, así
que hereda sus supuestos de signo. Además:

- `k = surface.ni * k0` (`stratton_chu.py:149`, `kirchhoff.py:95`) se vuelve
  negativo para el espejo (`ni = −n0`); `exp(i k R)` pasaría a onda entrante.
  Necesita `|ni|` con la orientación de propagación explícita.
- `PhysicalOpticsCurrents` usa `H = ni (u_i × E)` con coeficientes de
  **transmisión** (`currents.py:92`). Para un espejo hay que usar los de
  **reflexión** (`FresnelOvoid.reflection_coefficients`, `fresnel.py:107`) y la
  dirección reflejada.
- Aun así el solver de Franz es primero-principios dado `(J, M)`: es la vara de
  medir correcta una vez que la geometría y las corrientes se generalicen. En
  esta sesión se comprobó que, para el caso `z0>0>zi`, el solver **sí** enfoca
  en `z = zi < 0` con la amplitud canónica, mientras el paquete daba cero —
  confirmando que el error vivía en el paquete, no en la física.

## Diseño propuesto

1. **Clase `SignedDistance`** para `l0`, `li`: encapsula magnitud + signo por
   el sentido de recorrido en `z`. La condición estigmática pasa a ser la única
   fórmula de suma, `n0 l0 + ni li = −n0 z0 + ni zi`.
2. **Rederivar la geometría de rayos** (ángulos meridionales, cosenos de
   incidencia, orientación de normal, factor `A`, test de validez) a partir de
   las distancias con signo y de la dirección física de la luz, de modo que
   objeto/imagen virtual queden `valid` y con ángulos correctos.
3. **Fresnel según el régimen**: transmisión (`t_s, t_p`) para refracción,
   reflexión (`r_s, r_p`) para espejo (`ni = −n0`). El operador de
   transferencia elige según el índice.
4. **Propagadores y mapeo de pupila** con `|zi|` y `|cos(alpha_i)|`, sin
   guardas de signo que anulen el integrando.
5. **Generalizar el solver de referencia** (corrientes de reflexión, signo de
   `k`) para poder **validar contra Maxwell** los casos virtuales y de espejo.

## TODO

### Objetos e imágenes virtuales (refracción, mismo signo, `|n0| ≠ |ni|`)
- [ ] Clase `SignedDistance` y condición estigmática única.
- [ ] Geometría de rayos correcta para `z0>0` (objeto virtual) y `zi<0`
  (imagen virtual): normal, cosenos, `valid`, factor `A`.
- [ ] Mapeo de pupila y propagación con `|cos(alpha_i)|` y `|zi|`.
- [ ] `transfer.paraxial_channel_weights` con la forma con signo de `A'`.
- [ ] `system.relay_throughput` con `|cos_ai|`.
- [ ] Mensaje de `grazing_radius` corregido (no es «otro óvalo»).
- [ ] Validación contra el solver de referencia para los cuatro signos.

### Espejos (reflexión, `ni = −n0`)
- [ ] Operador de transferencia con coeficientes de reflexión `r_s, r_p`.
- [ ] Solver de referencia con corrientes de reflexión y `k = |ni| k0`.
- [ ] Casos de prueba: espejo elíptico (`l0+li=const`) e hiperbólico
  (`l0−li=const`), validados contra Maxwell.

### Pruebas
- [ ] Añadir sistemas de mismo signo y de espejo a la suite parametrizada de
  `tests/test_geometry_and_transfer.py`.
- [ ] Reproducir el campo focal exacto (Franz) para virtual y espejo, en
  amplitud, fase y forma.
