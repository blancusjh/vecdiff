# Notas: formalismos para medir la aberración vectorial

*Borrador para discusión (2026-07-04). Cuatro formalismos, sus ventajas y una
recomendación para la tesis.*

## 1. Coeficientes de Pauli / diatenuación–retardancia (Chipman)

Expandir la pupila de Jones $J = a_0\sigma_0 + a_1\sigma_1 + a_2\sigma_2 +
a_3\sigma_3$ y describir cada $a_k(r,\varphi)$ con pocos términos de bajo
orden (pistón, tilt, desenfoque de diatenuación y de retardancia).

- **Ventajas:** caracteriza el *operador* (independiente de la entrada);
  separa los mecanismos físicos — anisotropía de amplitud (diatenuación) vs.
  de fase (retardancia); pocos números con significado tipo Seidel. Para
  nuestro dioptrio dieléctrico colapsa a **dos funciones radiales**:
  $a_0 = t_+$, $(a_1,a_2) = t_-(\cos2\varphi,\sin2\varphi)$, retardancia
  $\equiv 0$; y conecta *exactamente* con el foco:
  $f_{\rm cross} = \langle(1-\sqrt{1-D^2})/2\rangle_w$,
  $A \simeq \kappa D_2 a^2/2$ (cuaderno 04).
- **Límites:** la serie de potencias se degrada cerca del rasante ($t_-$ deja
  de ser $\sim r^2$); requiere elegir una base transversal 2D.

## 2. Zernikes del campo residual (por componente)

Fijar una entrada, formar $E_{\rm out} - E_{\rm escalar}$ y expandir cada
componente cartesiana en Zernikes escalares.

- **Ventajas:** usa la maquinaria estándar de aberraciones; para nuestro
  sistema simétrico es sin pérdida y cae entero en la familia de
  **astigmatismo de amplitud** $m=\pm2$ ($Z_2^2, Z_4^2, \dots$), con
  coeficientes = proyecciones radiales de $t_-(r)$ sobre $R_n^2$. Ideal para
  *comunicar* en el lenguaje que todo óptico lee.
- **Límites:** describe el *estado* (una columna de $J$, depende de la
  entrada); los coeficientes por componente **no son covariantes ante
  rotaciones** (se mezclan al rotar los ejes). La corrección covariante son
  los *orientation Zernike polynomials* (Ruoff & Totzeck, 2009) o los
  Zernikes vectoriales (Zhao & Burge) — más maquinaria, valen la pena solo
  para sistemas sin simetría de revolución.
- **Ojo:** no normalizar el vector de polarización a módulo 1 — en un
  dieléctrico ($t_p,t_s$ reales) *toda* la aberración es de amplitud;
  normalizar mata el término dominante.

## 3. Pupila de Mueller (dominio de Stokes)

- **Ventajas:** es lo que mide un polarímetro real; maneja despolarización.
- **Límites:** 16 funciones reales, redundante para una superficie
  determinista como la nuestra. Descartado por ahora.

## 4. Polarization ray tracing 3×3 (Yun–Crabtree–Chipman, 2011)

Jones generalizado a matrices $P(r,\varphi)$ que actúan sobre campos 3D en la
esfera de salida.

- **Ventajas:** el único marco correcto a NA alta donde $E_z$ pesa; sin base
  transversal privilegiada; la retardancia se define contra el transporte
  paralelo (separa la fase geométrica).
- **Límites:** más pesado; relevante cuando el análisis incluya
  `vecdiff.longitudinal` de lleno (dependencia axial, sistemas no simétricos).

## Recomendación

1. **Formalismo primario: (1).** Para un dieléctrico con simetría de
   revolución es exacto, mínimo ($D(r)$, $D_2$, $D_{\rm rms}$) y ya tenemos
   los puentes cuantitativos pupila → deformación del punto focal.
2. **Capa fina de (2)** encima: los coeficientes Zernike $m=2$ de $t_-$
   (proyección 1D sobre $R_2^2, R_4^2, R_6^2$), para traducir el resultado a
   lenguaje estándar y cuantificar dónde el cuadro $D_2 r^2$ se rompe cerca
   del rasante. *Pendiente de implementar en el cuaderno 04.*
3. **(4) queda señalado** para el capítulo de $E_z$/NA alta.

## Referencias

- J. P. McGuire, R. A. Chipman, *Polarization aberrations. 1. Rotationally
  symmetric optical systems*, JOSA A **7**, 1614 (1990).
- R. A. Chipman, W.-S. T. Lam, G. Young, *Polarized Light and Optical
  Systems*, CRC Press (2019) — caps. de aberraciones de polarización.
- J. Ruoff, M. Totzeck, *Orientation Zernike polynomials: a useful way to
  describe the polarization effects of optical imaging systems*,
  J. Micro/Nanolith. MEMS MOEMS **8**, 031404 (2009).
- C. Zhao, J. H. Burge, *Orthonormal vector polynomials in a unit circle*,
  Opt. Express **15**, 18014 (2007).
- G. Yun, K. Crabtree, R. A. Chipman, *Three-dimensional polarization
  ray-tracing calculus I*, Appl. Opt. **50**, 2855 (2011).
