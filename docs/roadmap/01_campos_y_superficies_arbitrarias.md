# Campos y superficies arbitrarias

**Objetivo:** que `vecdiff` propague un campo de entrada arbitrario
(dependencia azimutal y polarización cualesquiera) a través de una superficie
arbitraria (sag cualquiera, no necesariamente de revolución), conservando los
caminos rápidos actuales como casos particulares.

## Límites actuales

**Campos.** El núcleo de Hankel exacto solo acepta pupilas con simetría axial:
`propagate_to_focal_plane_through_diopter` (`vecdiff/propagation.py:242`)
selecciona por `field.symmetry ∈ {circular, polar, cartesian}` y tiene los
órdenes cableados — `m = 0, 2` en las bases cartesiana y circular, `m = 1` en
la polar. Con `symmetry is None` levanta `NotImplementedError`. Todo lo demás
(vórtices, pupilas con estructura en φ, máscaras arbitrarias) debe pasar por
la rama FFT, que muestrea en cartesiano y pierde la ventaja radial.

**Superficies.** Existe una sola superficie, `CartesianSurface`
(`vecdiff/CartesianSurfaces.py`): el óvalo estigmático definido por
`(n0, ni, z0, zi)` vía los parámetros G, O, T, S. Es de revolución y los
coeficientes son transmisión dieléctrica real (sin recubrimientos ni
birrefringencia; ya hay `rs`/`rp` para el balance energético, pero no una rama
reflejada propagable). El encadenado de superficies se hace a mano en el
ejemplo (`examples/two_diopter_imaging.py:89`), no hay objeto sistema. Como el
óvalo es estigmático por construcción, la fase de camino óptico de la
superficie nunca se necesitó explícitamente: para superficies arbitrarias sí.

**Geometría del campo.** `Grid.reference` ya distingue esfera de referencia de
plano tangente, y los propagadores despachan sobre él. Pero la rama de entrada
sobre plano solo está implementada para mallas polares
(`vecdiff/propagation.py`, `_tangent_plane_to_incident_sphere`), y no hay
soporte para dar el campo sobre el plano objeto real en `z0` (que es lo que
suponen implícitamente los ejemplos de imagen encadenada: aplican el operador
de pupila sobre radios de plano que no son radios de pupila).

## TODO

### Campos arbitrarios
- [ ] Descomposición en armónicos azimutales: expandir la pupila en
  `exp(i m φ)` y propagar cada `m` con los órdenes de Hankel que le
  correspondan, en vez de cablear `m = 0, 1, 2`.
- [ ] Reemplazar el despacho por `field.symmetry` por un despacho por
  contenido armónico; `symmetry` pasa a ser un atajo (un solo armónico).
- [ ] Constructores de pupila de primera clase: vórtices de carga arbitraria,
  haces radial/azimutal con carga, Laguerre-Gauss y pupila de Jones
  `J(r, φ)` genérica.
- [ ] `Ez` para el camino armónico: `vecdiff/longitudinal.py` resuelve
  transversalidad por espectro angular en malla cartesiana; falta la versión
  radial/armónica.
- [ ] `Grid`: soportar malla radial no uniforme (cuadratura) y un contenedor
  de coeficientes armónicos junto a `type ∈ {cartesian, polar}`.

### Superficies arbitrarias
- [ ] Protocolo `Surface`: `sag(x, y)` (o `sag(rho)` cuando sea de revolución),
  `normal(...)` y marco local s/p; `CartesianSurface` lo implementa sin
  cambiar su API. Buena parte ya existe para el óvalo:
  `CartesianSurface.sag(r)`, `ray_geometry(r)` y `RayGeometry.local_frame`
  (`vecdiff/geometry.py`); falta generalizarlo a sag no de revolución.
- [ ] Implementaciones: esfera, cónica con asfericidad, plano y sag *freeform*
  (Zernike o spline) — estas últimas rompen la parametrización por `rho` y
  exigen normales de `∂z/∂x, ∂z/∂y`.
- [x] ~~Cerrar el TODO viejo de `vecdiff/fresnel.py` ("calculo de las
  bases")~~ — hecho: `RayGeometry.local_frame(phi)` devuelve `(ŝ, p̂₀, p̂ᵢ)`
  explícitos y el operador de `vecdiff/transfer.py` es puntual en ese marco,
  con `t±` como especialización a revolución. Falta el proveedor de marcos
  para superficies sin simetría de revolución.
- [ ] Operador de interfaz como Jones genérico: añadir reflexión (`rs`, `rp`),
  índice complejo (absorbente/metálico) y, si aparece la necesidad, apilado
  multicapa y birrefringencia.
- [x] ~~Factor de oblicuidad~~ — hecho para el óvalo: el factor geométrico
  `A(Q)` de conservación de flujo, la proyección meridional y el jacobiano del
  mapeo de pupila están en `vecdiff/transfer.py` y `vecdiff/pupil_mapping.py`,
  arbitrados contra la referencia de Maxwell (`examples/reference_vs_model.py`).
- [ ] Fase de camino óptico derivada del sag, para superficies no estigmáticas
  (el óvalo queda como el caso de fase nula).
- [ ] Objeto sistema `System([surface, ...])` que encadene superficies y
  aperturas, con el ejemplo de dos dioptrios reescrito sobre él.

### Numérica
- [ ] `HankelTransform`: hoy es Simpson en un bucle por cada `q`
  (`vecdiff/hankel.py:25`), coste O(N·M). Precomputar matrices de kernel por
  orden y vectorizar sobre órdenes y armónicos.
- [ ] `HT_N` tiene la malla `q` cableada (`q = linspace(1e-3, 10, 500)`):
  derivarla del muestreo de entrada.
- [x] ~~Tests de consistencia Hankel↔FFT~~ — hecho:
  `tests/test_geometry_and_transfer.py::test_hankel_and_fft_branches_agree`
  (coinciden a 7e-6 y convergen con el muestreo). Falta extenderlo a cada
  generalización armónica.

## Orden sugerido

1. Protocolo `Surface` + bases s/p explícitas (habilita todo lo demás y no
   cambia resultados).
2. Descomposición armónica en el camino de Hankel, con los tests de
   consistencia contra FFT.
3. Superficies no estigmáticas (fase de sag) y operador de Jones extendido.
4. `System` y limpieza de los ejemplos.

## Fuera de alcance por ahora

Trazado de rayos, dispersión cromática, campos no monocromáticos y
propagación en medios inhomogéneos.

## Cerrado por la corrección de la transferencia

La matriz de transferencia G → G′ y el factor geométrico de conservación de
energía ya están implementados y arbitrados contra un campo de Maxwell exacto
(integral de Franz/Stratton–Chu, `vecdiff/reference/`). Lo que queda anotado
como deuda de ese trabajo:

- [ ] **BEM riguroso.** La referencia usa corrientes de óptica física (reparto
  de Fresnel local), la misma hipótesis que el desarrollo analítico. Un solve
  de contorno Müller/PMCHWT sobre cuerpo de revolución la eliminaría; la
  interfaz ya está separada (`vecdiff/reference/currents.py`,
  `SurfaceCurrents`), así que es sustituir el proveedor de corrientes.
- [ ] **Sec. 12 del documento.** El mapeo perspectivo `r_P = |z₀| tan α₀` con
  factores `1/|cos α₀|` y `|cos αᵢ|` es correcto como relación de flujo entre
  plano y esfera, pero **no** es el cambio de variable de la transformada al
  plano focal: esa es una reducción de Debye y su jacobiano es el del mapeo
  seno. Medido contra la referencia a NA_i = 0.91, el mapeo perspectivo da
  14 % de error en amplitud y el seno 2e-8. Conviene corregir el documento.
- [ ] **Ejemplos de imagen encadenada.** `two_diopter_imaging`,
  `resolution_inversion` y `lithography` aplican el operador de pupila sobre
  coordenadas de plano que no son radios de pupila. Hoy no truncan (el aviso
  de apertura no salta), pero el modelo mezcla dos geometrías; se arregla con
  el soporte de entrada sobre plano objeto de la Fase 3.
