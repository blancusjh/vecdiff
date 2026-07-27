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
`(n0, ni, z0, zi)` vía los parámetros G, O, T, S. Es de revolución, se
parametriza por un único `rho`, y expone `z`, `dz`, `r` — no una normal 2D.
`FresnelOvoid._cosines` (`vecdiff/fresnel.py:62`) depende de esa
parametrización, y los coeficientes son transmisión dieléctrica real
(sin reflexión, sin `n` compleja, sin recubrimientos ni birrefringencia).
El encadenado de superficies se hace a mano en el ejemplo
(`examples/two_diopter_imaging.py:89`), no hay objeto sistema. Como el óvalo es
estigmático por construcción, la fase de camino óptico de la superficie nunca
se necesitó explícitamente: para superficies arbitrarias sí.

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
  cambiar su API.
- [ ] Implementaciones: esfera, cónica con asfericidad, plano y sag *freeform*
  (Zernike o spline) — estas últimas rompen la parametrización por `rho` y
  exigen normales de `∂z/∂x, ∂z/∂y`.
- [ ] Cerrar el TODO viejo de `vecdiff/fresnel.py:115` ("calculo de las
  bases"): devolver los versores s/p explícitos, que es lo que permite el
  operador fuera de eje y sin simetría.
- [ ] Operador de interfaz como Jones genérico: añadir reflexión (`rs`, `rp`),
  índice complejo (absorbente/metálico) y, si aparece la necesidad, apilado
  multicapa y birrefringencia.
- [ ] Fase de camino óptico y factor de oblicuidad derivados del sag, para
  superficies no estigmáticas (el óvalo queda como el caso de fase nula).
- [ ] Objeto sistema `System([surface, ...])` que encadene superficies y
  aperturas, con el ejemplo de dos dioptrios reescrito sobre él.

### Numérica
- [ ] `HankelTransform`: hoy es Simpson en un bucle por cada `q`
  (`vecdiff/hankel.py:25`), coste O(N·M). Precomputar matrices de kernel por
  orden y vectorizar sobre órdenes y armónicos.
- [ ] `HT_N` tiene la malla `q` cableada (`q = linspace(1e-3, 10, 500)`):
  derivarla del muestreo de entrada.
- [ ] Tests de consistencia armónico↔FFT: para cada generalización, verificar
  que ambos caminos coinciden y que el caso simétrico reproduce los
  resultados actuales a precisión de máquina.

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
