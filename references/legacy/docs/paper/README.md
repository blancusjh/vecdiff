# Documento

`refraccion_sistema_estigmatico.tex` — derivación de la transferencia vectorial
y energética del dioptrio estigmático simple, en la versión que el paquete
implementa y verifica.

Cada resultado del documento tiene su contraparte ejecutable:

| Sección | Verificado por |
| --- | --- |
| Parametrización por `r`, condición estigmática, Snell | `tests/test_geometry_and_transfer.py` |
| Factor geométrico `A(Q)`, balance de flujo | `tests/test_geometry_and_transfer.py` |
| Transversalidad `λ_z = λ_r tan αᵢ` | `test_pupil_fields_are_divergence_free` |
| Reducción de Debye, mapeo seno vs. tangente | `examples/reference_vs_model.py` |
| Versión escalar (canal `s`) | `examples/scalar_reference_check.py` |
| Límites de apertura (rasancia / ángulo crítico) | `examples/aperture_limits.py` |
| Árbitro vectorial (Franz/Stratton–Chu) | `vecdiff/reference/stratton_chu.py` |
| Árbitro escalar (Helmholtz–Kirchhoff) | `vecdiff/reference/kirchhoff.py` |

## Compilar

Con [tectonic](https://tectonic-typesetting.github.io), que resuelve los
paquetes solo:

```bash
tectonic -X compile docs/paper/refraccion_sistema_estigmatico.tex
```

o bien `latexmk -pdf docs/paper/refraccion_sistema_estigmatico.tex`.
