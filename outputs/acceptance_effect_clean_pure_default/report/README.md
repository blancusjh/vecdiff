# Estudio Breve: Contribución No Escalar en Intensidad

## Objetivo
Evaluar cómo cambia la diferencia entre el caso vectorial total y el escalar, enfocándonos en:

- Intensidad central (`I(0)`).
- Diferencia pura `ΔI = I_total - I_scalar`.
- Tendencia con ángulo de aceptación (`alpha`) y con `ni` (proxy de NA efectiva junto con alpha).

## Métricas usadas
- `F_c = I_total(0) / I_scalar(0)`.
- `D_c = 1 - F_c` (déficit central relativo del vectorial respecto al escalar).
- `rel_l1_delta = ∫|ΔI| q dq / ∫I_scalar q dq`.

Interpretación rápida:
- `F_c < 1`: el escalar predice mayor intensidad central que el vectorial.
- `D_c` alto: la corrección vectorial en el centro es importante.

## Figuras seleccionadas
### Tendencias globales
- Paraxial: ![paraxial central](figures/central_factor_vs_alpha_paraxial.png)
- Exacto: ![exact central](figures/central_factor_vs_alpha_exact.png)

- Paraxial (heatmaps): ![paraxial heat](figures/heatmaps_metrics_paraxial.png)
- Exacto (heatmaps): ![exact heat](figures/heatmaps_metrics_exact.png)

### Caso representativo de alta contribución (`ni=2.6`, `alpha=40°`)
- Paraxial (lineal): ![case parax](figures/case_ni2p6_alpha40_paraxial_linear.png)
- Exacto (lineal): ![case exact](figures/case_ni2p6_alpha40_exact_linear.png)
- Exacto (log): ![case exact log](figures/case_ni2p6_alpha40_exact_log.png)

## Hallazgos principales
1. En toda la grilla analizada, el centro del campo total vectorial es menor que el escalar (`F_c < 1` en los casos dominantes).
2. La diferencia no escalar crece al movernos a condiciones de mayor apertura/índice, y se hace claramente visible en `D_c` y `rel_l1_delta`.
3. El modelo exacto amplifica la discrepancia respecto al paraxial en varios casos, especialmente a `alpha` altos.

## Números clave (comparación paraxial vs exacto)
Top aumentos en déficit central `ΔD_c = D_c(exacto) - D_c(paraxial)`:

- `ni=1.8, alpha=40°`: `+0.2115`
- `ni=2.2, alpha=40°`: `+0.1485`
- `ni=1.8, alpha=30°`: `+0.1216`

Máximo déficit central encontrado en exacto dentro de esta grilla:

- `ni=2.6, alpha=40°`: `D_c = 0.7540` (`F_c = 0.2460`), con `rel_l1_delta = 0.7602`.

## Conclusiones técnicas
1. El modelo escalar sobreestima la intensidad central cuando el régimen sale de paraxialidad moderada.
2. La magnitud de la corrección vectorial es suficientemente grande como para afectar conclusiones experimentales si se usa solo escalar.
3. Tiene sentido estudiar no solo `alpha`, sino una parametrización más general en NA, porque el efecto depende de la combinación de geometría y contraste de índices.
4. Para análisis de diseño, conviene reportar siempre el par `(F_c, D_c)` junto con una métrica energética de `ΔI`.

## Próximos pasos sugeridos
1. Barrido explícito en NA (no solo `alpha`) con la misma métrica central.
2. Repetir para varios perfiles incidentes (`p=0,2,4,6`) y comparar sensibilidad.
3. Construir mapa de región “segura para escalar” usando umbral en `D_c` (por ejemplo `D_c < 0.05`).
