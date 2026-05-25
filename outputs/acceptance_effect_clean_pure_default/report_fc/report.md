# Reporte de Resultados: Caso Vectorial vs Escalar

## 1) Qué se midió
Se comparó el caso **vectorial total** frente al **escalar** para tres polarizaciones (`circular`, `cartesian`, `polar`) y dos modelos (`paraxial`, `exact`).

Métrica principal:
- `F_c = I_total(0) / I_scalar(0)`

Interpretación:
- `F_c < 1` implica que la intensidad central vectorial es menor que la escalar.

Métrica de tamaño de mancha:
- `spot_ratio = w_total / w_scalar`
- `w` se estimó con **EE50** (radio que encierra el 50% de energía), más robusto que FWHM en perfiles oscilatorios.

---

## 2) Conclusión principal sobre intensidad central
La intensidad central del caso vectorial es consistentemente más modesta.

Resultados globales (72 casos):
- `F_c` promedio: **0.387**
- `F_c` mínimo: **0.246**
- `F_c` máximo: **0.589**

Equivalente en déficit central (`1 - F_c`):
- déficit promedio: **61.3%**
- déficit máximo: **75.4%**

Esto cuantifica que el escalar tiende a sobreestimar el pico central frente al vectorial en este régimen.

---

## 3) Diferencia entre modelo paraxial y exacto
Promedios por modelo:
- `paraxial`: `F_c` promedio = **0.411**
- `exact`: `F_c` promedio = **0.364**

Interpretación:
- El modelo **exacto** predice una caída central aún mayor (más severa) que el paraxial.

Casos de mayor caída observada:
- `circular/exact, ni=2.6, alpha=40°`: `F_c = 0.246`
- `cartesian/exact, ni=2.6, alpha=40°`: `F_c = 0.246`

---

## 4) Cómo cambia con el ángulo de aceptación
Promedio global de `F_c` por ángulo:
- `alpha=10°`: `F_c ≈ 0.400`
- `alpha=20°`: `F_c ≈ 0.391`
- `alpha=30°`: `F_c ≈ 0.378`
- `alpha=40°`: `F_c ≈ 0.381`

Lectura:
- La diferencia central es grande en todo el rango y tiende a empeorar al pasar a aperturas altas (en especial en combinaciones `exact` + `ni` alto).

---

## 5) Spot size (vectorial vs escalar)
### Resumen
Para `circular` y `cartesian`, la comparación promedio conjunta da una mancha vectorial ligeramente mayor en términos agregados, aunque con dependencia de modelo:

- `circular/paraxial`: `spot_ratio` promedio = **1.0569**
- `cartesian/paraxial`: `spot_ratio` promedio = **1.0569**
- `circular/exact`: `spot_ratio` promedio = **0.9936** (cerca de 1)
- `cartesian/exact`: `spot_ratio` promedio = **0.9936** (cerca de 1)
- `polar/paraxial`: `spot_ratio` promedio = **0.9914`
- `polar/exact`: `spot_ratio` promedio = **0.9961`

Lectura física práctica:
- En paraxial (circular/cartesiana), el ensanchamiento promedio del spot vectorial es claro.
- En exacto, el spot ratio queda muy cercano a la unidad y puede cambiar de lado según el caso; el efecto dominante sigue siendo la reducción del pico central (`F_c < 1`).

---

## 6) Relevancia de la diferencia
La diferencia no es cosmética:
- El pico central puede caer entre ~40% y ~75% según parámetros.
- Esto impacta directamente predicciones de brillo en foco, umbrales, y comparación con medidas experimentales.
- Usar solo escalar en estos regímenes puede sesgar conclusiones cuantitativas.

---

## 7) Mensaje final
1. **La intensidad central vectorial es sistemáticamente menor que la escalar** en este estudio.
2. **El modelo exacto acentúa** esa diferencia respecto al paraxial.
3. **El spot size tiende a ser mayor en promedio** para circular/cartesiana en paraxial; en exacto aparece casi neutro en promedio y dependiente del caso.
4. Para análisis robusto de diseño/interpretación experimental, conviene reportar al menos `F_c` (y opcionalmente `spot_ratio`) en lugar de asumir equivalencia escalar.
