# Reporte F_c por Polarización

Métrica principal: `F_c = I_total(0)/I_scalar(0)`.
Métrica adicional: `w_total/w_scalar` (comparación de spot size a media altura).

## Lectura
- `F_c < 1`: el total vectorial tiene menor intensidad central que el escalar.
- `F_c ≈ 1`: diferencia central baja.
- `w_total/w_scalar > 1`: el spot vectorial es más ancho que el escalar.
- `w_total/w_scalar < 1`: el spot vectorial es más estrecho.

## Casos con menor F_c (mayor caída central del total)
- pol=circular, model=exact, ni=2.60, alpha=40.0°, F_c=0.2460
- pol=cartesian, model=exact, ni=2.60, alpha=40.0°, F_c=0.2460
- pol=cartesian, model=exact, ni=2.60, alpha=30.0°, F_c=0.2713
- pol=circular, model=exact, ni=2.60, alpha=30.0°, F_c=0.2713
- pol=polar, model=paraxial, ni=2.60, alpha=30.0°, F_c=0.2838
- pol=polar, model=exact, ni=2.60, alpha=30.0°, F_c=0.2880

## Figuras
### circular | paraxial
![Fc-alpha](Fc_vs_alpha_circular_paraxial.png)
![Fc-heatmap](Fc_heatmap_circular_paraxial.png)

### circular | exact
![Fc-alpha](Fc_vs_alpha_circular_exact.png)
![Fc-heatmap](Fc_heatmap_circular_exact.png)

### polar | paraxial
![Fc-alpha](Fc_vs_alpha_polar_paraxial.png)
![Fc-heatmap](Fc_heatmap_polar_paraxial.png)

### polar | exact
![Fc-alpha](Fc_vs_alpha_polar_exact.png)
![Fc-heatmap](Fc_heatmap_polar_exact.png)

### cartesian | paraxial
![Fc-alpha](Fc_vs_alpha_cartesian_paraxial.png)
![Fc-heatmap](Fc_heatmap_cartesian_paraxial.png)

### cartesian | exact
![Fc-alpha](Fc_vs_alpha_cartesian_exact.png)
![Fc-heatmap](Fc_heatmap_cartesian_exact.png)
