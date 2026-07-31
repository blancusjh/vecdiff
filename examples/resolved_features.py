"""Which features survive vectorial imaging, and which ones scalar theory only thinks it resolves.

Scalar diffraction theory says whether a periodic feature is resolved depends on
its period and on the numerical aperture, and on nothing else.  Vectorial theory
says it also depends on how the feature is oriented with respect to the
illumination polarization, and near the resolution limit the difference is not a
correction: it is a factor of four.

The mechanism is interference geometry, not Fresnel transmission.  A periodic
feature of period ``p`` sends its two first orders into the image at ``+-theta`` with
``sin(theta) = lambda / (n p)``.  With the field polarized along the lines the
two orders stay parallel (TE, s) and interfere at full contrast.  Polarized
across them, the two electric vectors are tilted apart by ``2 theta`` (TM, p),
the fringe contrast falls as ``cos(2 theta)``, and a longitudinal component
``E_z`` carries off the difference.  At the coherent cutoff of an immersion
system with ``NA = 0.88`` in water that is ``cos(75.8 deg) = 0.24``.

Model
-----
Coherent imaging with a vectorial pupil.  The object spectrum is laid
on the pupil of the stigmatic exit surface, weighted by the transfer operator
of :mod:`vecdiff.transfer`, and transformed to the image.  The pupil coordinate
is the sine-mapped one the Debye reduction integrates over, so a spatial
frequency ``nu`` lands at ``sin(alpha_i) = lambda nu / n_i`` and the pupil edge
falls exactly on the coherent cutoff ``NA / lambda``.

The scalar reference keeps the *same* amplitude apodization and drops only the
polarization mixing, so the comparison isolates polarization instead of
confounding it with pupil weighting.

Run from the repository root::

    uv run python examples/resolved_features.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vecdiff import CartesianSurface
from vecdiff.propagation import transfer_weights_on_grid

from _output import example_output_dir, print_saved

π = np.pi

# ---------------------------------------------------------------------
# Optical configuration
# ---------------------------------------------------------------------

lam = 193e-6  # mm, ArF

# A high-aperture immersion configuration: a LuAG last element (n ~ 2.14) into
# water (n ~ 1.437).  A silica-to-air exit cannot reach this aperture -- a
# dense-to-rare stigmatic surface runs into the critical angle and saturates
# near NA 0.74 -- so the high-index element and the immersion medium are what
# make the regime reachable at all.
N_LUAG = 2.14
N_WATER = 1.437
LENS = dict(n0=N_LUAG, ni=N_WATER, z0=-42.0, zi=2.0)

SURFACE = CartesianSurface(**LENS)
APERTURE = SURFACE.aperture_limit

# The usable aperture sets the coherent cutoff; nothing here is asserted.
_edge = SURFACE.ray_geometry(np.array([APERTURE * (1.0 - 1e-6)]))
SIN_MAX = float(_edge.sin_ai[0])
NA = LENS["ni"] * SIN_MAX
NU_CUT = NA / lam                     # coherent cutoff, cycles per mm
PITCH_CUT = 1.0 / NU_CUT              # smallest pitch that reaches the pupil edge

out_dir = example_output_dir(__file__)


# ---------------------------------------------------------------------
# Vectorial pupil
# ---------------------------------------------------------------------

def _radius_from_sine(sin_ai, n_table=200_001):
    """Invert ``sin(alpha_i)(r)`` on the surface, returning the pupil radius ``r``."""
    r_table = np.linspace(0.0, APERTURE * (1.0 - 1e-9), n_table)
    s_table = SURFACE.ray_geometry(r_table).sin_ai
    order = np.argsort(s_table)
    return np.interp(np.clip(sin_ai, 0.0, s_table.max()), s_table[order], r_table[order])


_PUPIL_CACHE: dict = {}


def pupil_weights(NUX, NUY, *, geometry="full"):
    """Return ``(w_plus, w_minus, w_z, inside)`` on a spatial-frequency grid.

    ``w_plus`` and ``w_minus`` are the mean and half-difference of the radial
    and azimuthal transfer eigenvalues, including the sine-mapping Jacobian --
    exactly the pair the Cartesian form of the operator consumes.
    """
    # The weights depend only on the grid, and the sweep reuses one grid, so
    # rebuilding them per pitch would dominate the run time.
    key = (NUX.shape, float(NUX[0, 1] - NUX[0, 0]), float(NUY[1, 0] - NUY[0, 0]), geometry)
    cached = _PUPIL_CACHE.get(key)
    if cached is not None:
        return cached

    NU = np.hypot(NUX, NUY)
    sin_ai = lam * NU / LENS["ni"]
    inside = sin_ai <= SIN_MAX

    r = _radius_from_sine(np.where(inside, sin_ai, 0.0))
    lam_r, lam_phi, lam_z = transfer_weights_on_grid(r, SURFACE, geometry=geometry)

    geom = SURFACE.ray_geometry(r)
    if geometry == "none":
        jac = np.ones_like(geom.cos_ai)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            jac = np.where(geom.cos_ai > 0.0, 1.0 / geom.cos_ai, 0.0)

    zero = np.zeros_like(lam_r)
    result = (
        np.where(inside, 0.5 * (lam_r + lam_phi) * jac, zero),
        np.where(inside, 0.5 * (lam_r - lam_phi) * jac, zero),
        np.where(inside, lam_z * jac, zero),
        inside,
    )
    _PUPIL_CACHE[key] = result
    return result


def image_fields(obj, dx, polarization=(1.0, 0.0), *, model="vectorial", geometry="full"):
    """Return ``(Ex, Ey, Ez)`` in the image plane for an object in image coordinates.

    ``model`` is ``"vectorial"`` (the full operator), ``"scalar"`` (the same
    amplitude apodization with the polarization mixing removed) or ``"flat"``
    (an unapodized scalar pupil, the textbook reference).
    """
    ny, nx = obj.shape
    NUX, NUY = np.meshgrid(
        np.fft.fftfreq(nx, d=dx), np.fft.fftfreq(ny, d=dx), indexing="xy"
    )

    spectrum = np.fft.fft2(obj.astype(complex))
    px, py = polarization
    Sx, Sy = px * spectrum, py * spectrum

    if model == "flat":
        inside = np.hypot(NUX, NUY) <= NU_CUT
        return np.fft.ifft2(Sx * inside), np.fft.ifft2(Sy * inside), np.zeros_like(Sx)

    w_plus, w_minus, w_z, _ = pupil_weights(NUX, NUY, geometry=geometry)
    if model == "scalar":
        w_minus = np.zeros_like(w_minus)
        w_z = np.zeros_like(w_z)
    elif model != "vectorial":
        raise ValueError("model must be 'vectorial', 'scalar' or 'flat'.")

    phi = np.arctan2(NUY, NUX)
    c2, s2 = np.cos(2.0 * phi), np.sin(2.0 * phi)

    Px = (w_plus + w_minus * c2) * Sx + (w_minus * s2) * Sy
    Py = (w_minus * s2) * Sx + (w_plus - w_minus * c2) * Sy
    Pz = w_z * (np.cos(phi) * Sx + np.sin(phi) * Sy)

    return np.fft.ifft2(Px), np.fft.ifft2(Py), np.fft.ifft2(Pz)


# ---------------------------------------------------------------------
# Gratings and contrast
# ---------------------------------------------------------------------

def snap_pitch(pitch, dx):
    """Round a pitch to a whole number of samples.

    The grating has to be exactly periodic on the grid: otherwise the sampled
    duty cycle wobbles from pitch to pitch and the contrast curve picks up
    jitter that has nothing to do with the optics.
    """
    return max(int(round(pitch / dx)), 2) * dx


def grating(shape, dx, pitch, orientation):
    """Binary line/space grating with a 1:1 duty cycle, in image coordinates."""
    ny, nx = shape
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dx
    X, Y = np.meshgrid(x, y, indexing="xy")
    coord = X if orientation == "vertical" else Y
    return (np.mod(coord, pitch) < 0.5 * pitch).astype(float)


def contrast(intensity, dx, pitch, orientation):
    """Fringe contrast measured over the central few periods."""
    ny, nx = intensity.shape
    if orientation == "vertical":
        profile = intensity[ny // 2, :]
        axis = (np.arange(nx) - nx // 2) * dx
    else:
        profile = intensity[:, nx // 2]
        axis = (np.arange(ny) - ny // 2) * dx

    core = np.abs(axis) <= 2.5 * pitch
    if np.count_nonzero(core) < 8:
        core = np.ones_like(axis, dtype=bool)
    values = profile[core]
    hi, lo = float(values.max()), float(values.min())
    return (hi - lo) / (hi + lo) if hi + lo > 0 else 0.0


def total_intensity(fields):
    Ex, Ey, Ez = fields
    return np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2


# ---------------------------------------------------------------------
# A layout, not just a grating
# ---------------------------------------------------------------------

def feature_layout(shape, dx, unit):
    """A set of features of different shape and orientation, sized against ``unit``.

    ``unit`` is the coherent cutoff pitch, so every feature here sits within a
    factor of two of the resolution limit -- the regime where the vectorial
    answer parts company with the scalar one.

    Nothing about this layout needs to sit near the axis.  The pupil model is
    shift invariant, so a feature ten microns off axis is imaged by the same
    pupil as one on it; the aberration that made the old two-surface relay
    unusable for extended masks simply does not arise here.
    """
    ny, nx = shape
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dx
    X, Y = np.meshgrid(x, y, indexing="xy")
    T = np.zeros_like(X)
    groups = {}

    def rect(cx, cy, w, h):
        T[(np.abs(X - cx) <= 0.5 * w) & (np.abs(Y - cy) <= 0.5 * h)] = 1.0

    def disc(cx, cy, r):
        T[(X - cx) ** 2 + (Y - cy) ** 2 <= r * r] = 1.0

    # 1. Dense vertical lines: orders separate along x, so this group is TM
    #    under x-polarized illumination and is the one that suffers.
    p_dense = 1.05 * unit
    cx, cy = -9.0 * unit, 6.5 * unit
    for m in range(-2, 3):
        rect(cx + m * p_dense, cy, 0.5 * p_dense, 7.0 * unit)
    groups["líneas verticales (TM)"] = (cx, cy, 6.0 * unit, 4.5 * unit, "x")

    # 2. The same pitch rotated: TE, and it should survive untouched.
    cx, cy = 6.0 * unit, 6.5 * unit
    for m in range(-2, 3):
        rect(cx, cy + m * p_dense, 7.0 * unit, 0.5 * p_dense)
    groups["líneas horizontales (TE)"] = (cx, cy, 6.0 * unit, 4.5 * unit, "y")

    # 3. Contact array.
    p_contact = 1.7 * unit
    cx, cy = -9.0 * unit, -6.0 * unit
    for i in range(-1, 2):
        for j in range(-1, 2):
            disc(cx + i * p_contact, cy + j * p_contact, 0.34 * unit)
    groups["contactos"] = (cx, cy, 3.8 * unit, 3.8 * unit, "x")

    # 4. Elbows and line ends: the features that show corner rounding and
    #    pull-back, which is what a layout engineer actually looks at.
    cx, cy = 5.5 * unit, -6.0 * unit
    arm, wid = 5.0 * unit, 0.55 * unit
    rect(cx - 0.5 * arm + 0.5 * wid, cy + 2.0 * unit, arm, wid)
    rect(cx - 0.5 * arm + 0.5 * wid, cy + 2.0 * unit - 0.5 * arm + 0.5 * wid, wid, arm)
    rect(cx + 2.2 * unit, cy - 2.0 * unit, 3.2 * unit, wid)
    rect(cx + 2.2 * unit + 3.2 * unit + 1.3 * unit, cy - 2.0 * unit, 3.2 * unit, wid)
    groups["codos y extremos"] = (cx + 4.3 * unit, cy - 2.0 * unit, 9.0 * unit, 1.0 * unit, "x")

    return T, groups


def group_contrast(intensity, dx, box):
    """Contrast along the direction the group's features repeat.

    Measured on a cut through the group centre rather than over its whole box:
    isolated features sit on zero background, so a box-wide min/max reports 1.0
    whatever the optics did.  What matters is how deep the valley *between*
    neighbouring features is.
    """
    ny, nx = intensity.shape
    cx, cy, w, h, direction = box
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dx
    ix = np.abs(x - cx) <= 0.5 * w
    iy = np.abs(y - cy) <= 0.5 * h

    if direction == "x":
        profile = intensity[np.argmin(np.abs(y - cy)), ix]
    elif direction == "y":
        profile = intensity[iy, np.argmin(np.abs(x - cx))]
    else:
        patch = intensity[np.ix_(iy, ix)]
        row = patch[patch.shape[0] // 2, :]
        col = patch[:, patch.shape[1] // 2]
        profile = row if row.ptp() >= col.ptp() else col

    hi, lo = float(profile.max()), float(profile.min())
    return (hi - lo) / (hi + lo) if hi + lo > 0 else 0.0


# ---------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------

#: Below this fringe contrast a feature is not usefully resolved.  This is a
#: modulation criterion, not Rayleigh's: the orders sit inside the pupil at
#: every period swept here, so what fails is the modulation, not the collection.
CONTRAST_THRESHOLD = 0.5

ORIENTATIONS = {
    # x-polarized illumination.  Vertical lines run along y, across the
    # polarization: their orders separate along x and the field is p (TM).
    "vertical": dict(label="líneas ⊥ a la polarización (TM)", key="TM"),
    # Horizontal lines run along x, with the polarization: the orders separate
    # along y and the field stays s (TE).
    "horizontal": dict(label="líneas ∥ a la polarización (TE)", key="TE"),
}


def run_pitch_sweep(n=1024, oversample=8.0):
    dx = PITCH_CUT / oversample
    shape = (n, n)
    factors = np.linspace(1.005, 3.0, 28)

    results = {}
    for orientation, meta in ORIENTATIONS.items():
        rows = {"pitch": [], "flat": [], "scalar": [], "vectorial": []}
        for f in factors:
            pitch = snap_pitch(f * PITCH_CUT, dx)
            obj = grating(shape, dx, pitch, orientation)
            for model in ("flat", "scalar", "vectorial"):
                I = total_intensity(image_fields(obj, dx, model=model))
                rows[model].append(contrast(I, dx, pitch, orientation))
            rows["pitch"].append(pitch / PITCH_CUT)
        results[meta["key"]] = {k: np.asarray(v) for k, v in rows.items()}
    return results, dx, shape


def resolved_limit(factors, contrasts):
    """Smallest period factor whose contrast clears the resolution threshold."""
    ok = contrasts >= CONTRAST_THRESHOLD
    if not np.any(ok):
        return np.nan
    idx = int(np.flatnonzero(ok)[0])
    if idx == 0:
        return float(factors[0])
    f0, f1 = factors[idx - 1], factors[idx]
    c0, c1 = contrasts[idx - 1], contrasts[idx]
    return float(f0 + (f1 - f0) * (CONTRAST_THRESHOLD - c0) / (c1 - c0))


def main():
    print(f"superficie de salida: LuAG (n={N_LUAG}) → agua (n={N_WATER}), "
          f"z0={LENS['z0']} mm, zi={LENS['zi']} mm")
    print(f"apertura útil r = {APERTURE:.4f} mm,  sin(alpha_i) max = {SIN_MAX:.4f}")
    print(f"NA = {NA:.4f},  corte coherente = {NU_CUT:.1f} ciclos/mm,  "
          f"paso de corte = {PITCH_CUT * 1e6:.1f} nm = {PITCH_CUT / lam:.3f} λ")
    sin_th = min(NA / LENS["ni"], 1.0)
    print(f"a ese paso los órdenes salen a ±{np.degrees(np.arcsin(sin_th)):.1f}°, "
          f"y cos(2θ) = {np.cos(2 * np.arcsin(sin_th)):+.4f}")
    print()

    results, dx, shape = run_pitch_sweep()

    print("Período mínimo resuelto (contraste ≥ "
          f"{CONTRAST_THRESHOLD:.2f}), en unidades del paso de corte:")
    limits = {}
    for key, data in results.items():
        limits[key] = {
            m: resolved_limit(data["pitch"], data[m]) for m in ("scalar", "vectorial")
        }
        ratio = limits[key]["vectorial"] / limits[key]["scalar"]
        print(f"   {key}:  escalar {limits[key]['scalar']:.3f}   "
              f"vectorial {limits[key]['vectorial']:.3f}   ({ratio:.2f}x)")
    print()

    print("Contraste cerca del corte:")
    print(f"   {'paso/corte':>10}  {'TM escalar':>11}  {'TM vect.':>10}  "
          f"{'TE vect.':>10}  {'TM vect/esc':>12}")
    for i, f in enumerate(results["TM"]["pitch"]):
        if f > 1.7 or i % 3:
            continue
        ts, tv = results["TM"]["scalar"][i], results["TM"]["vectorial"][i]
        ev = results["TE"]["vectorial"][i]
        print(f"   {f:>10.3f}  {ts:>11.4f}  {tv:>10.4f}  {ev:>10.4f}  {tv / ts:>12.3f}")
    print()

    tm = results["TM"]
    window = (tm["scalar"] >= CONTRAST_THRESHOLD) & (tm["vectorial"] < CONTRAST_THRESHOLD)
    if np.any(window):
        lo = tm["pitch"][window].min() * PITCH_CUT * 1e6
        hi = tm["pitch"][window].max() * PITCH_CUT * 1e6
        print("Ventana donde la teoría escalar resuelve y la vectorial NO (TM): "
              f"período {lo:.0f}–{hi:.0f} nm")
    else:
        print("No hay ventana en la que escalar resuelva y vectorial no.")
    print()

    _figure_contrast(results)
    _figure_pattern()
    _figure_aerial(dx, shape)
    _figure_mechanism(dx, shape)


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def _figure_contrast(results):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)
    for ax, (key, data) in zip(axes, results.items()):
        p_nm = data["pitch"] * PITCH_CUT * 1e6
        ax.plot(p_nm, data["flat"], "^-", ms=3.5, lw=1.2, alpha=0.6,
                label="escalar, pupila plana")
        ax.plot(p_nm, data["scalar"], "o-", ms=3.5, lw=1.6,
                label="escalar (misma apodización)")
        ax.plot(p_nm, data["vectorial"], "s-", ms=3.5, lw=1.6, label="vectorial")
        ax.axhline(CONTRAST_THRESHOLD, color="k", ls=":", lw=1.2)
        ax.text(p_nm[-1], CONTRAST_THRESHOLD, " umbral", va="bottom", ha="right", fontsize=8)
        if key == "TM":
            band = (data["scalar"] >= CONTRAST_THRESHOLD) & (data["vectorial"] < CONTRAST_THRESHOLD)
            if np.any(band):
                ax.axvspan(p_nm[band].min(), p_nm[band].max(), color="crimson",
                           alpha=0.15, label="escalar resuelve,\nvectorial no")
        label = next(m["label"] for m in ORIENTATIONS.values() if m["key"] == key)
        ax.set_title(label)
        ax.set_xlabel("período [nm]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("contraste de franjas")
    axes[0].set_ylim(-0.02, 1.05)
    fig.suptitle(
        f"Contraste de imagen, NA={NA:.2f} en agua, λ=193 nm  |  "
        f"paso de corte {PITCH_CUT * 1e6:.0f} nm",
        fontsize=11,
    )
    fig.tight_layout()
    path = out_dir / "resolved_features_contrast.png"
    fig.savefig(path, dpi=150)
    print_saved(path)
    plt.close(fig)


def _figure_pattern(n=1536, oversample=10.0):
    """The layout printed by both theories, side by side."""
    dx = PITCH_CUT / oversample
    obj, groups = feature_layout((n, n), dx, PITCH_CUT)

    images = {}
    for model in ("scalar", "vectorial"):
        images[model] = total_intensity(image_fields(obj, dx, model=model))

    peak = max(I.max() for I in images.values())
    images = {k: I / peak for k, I in images.items()}

    print("Contraste por grupo de features:")
    print(f"   {'grupo':<28} {'escalar':>9} {'vectorial':>10} {'razón':>7}")
    rows = []
    for name, box in groups.items():
        cs = group_contrast(images["scalar"], dx, box)
        cv = group_contrast(images["vectorial"], dx, box)
        rows.append((name, cs, cv))
        print(f"   {name:<28} {cs:>9.4f} {cv:>10.4f} {cv / cs if cs else 0:>7.3f}")
    print()

    half = n // 2 * dx / lam
    extent = [-half, half, -half, half]
    fig, axes = plt.subplots(1, 4, figsize=(21.0, 5.6))

    axes[0].imshow(obj, extent=extent, origin="lower", cmap="gray")
    axes[0].set_title("objeto", fontsize=11)

    for ax, model, title in (
        (axes[1], "scalar", "imagen escalar"),
        (axes[2], "vectorial", "imagen vectorial"),
    ):
        ax.imshow(images[model], extent=extent, origin="lower", cmap="inferno",
                  vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=11)

    diff = images["vectorial"] - images["scalar"]
    lim = float(np.abs(diff).max())
    im = axes[3].imshow(diff, extent=extent, origin="lower", cmap="RdBu_r",
                        vmin=-lim, vmax=lim)
    axes[3].set_title("vectorial − escalar", fontsize=11)
    fig.colorbar(im, ax=axes[3], fraction=0.046)

    for box in groups.values():
        cx, cy, w, h, _ = box
        for ax in axes:
            ax.add_patch(plt.Rectangle(
                ((cx - 0.5 * w) / lam, (cy - 0.5 * h) / lam), w / lam, h / lam,
                fill=False, ec="lime", lw=0.9, ls="--", alpha=0.8))

    span = max(
        max(abs(cx) + 0.5 * w for cx, _, w, _, _ in groups.values()),
        max(abs(cy) + 0.5 * h for _, cy, _, h, _ in groups.values()),
    ) / lam * 1.12
    for ax in axes:
        ax.set_xlabel(r"$x/\lambda$")
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_aspect("equal")
    axes[0].set_ylabel(r"$y/\lambda$")

    summary = "   ".join(f"{name}: {cs:.2f} → {cv:.2f}" for name, cs, cv in rows)

    fig.suptitle(
        f"Conjunto de features imagenado a NA={NA:.2f} en agua, λ=193 nm, iluminación $\\hat{{x}}$  |  "
        f"paso de corte {PITCH_CUT * 1e6:.0f} nm.  "
        "Misma apodización en ambas imágenes: la única diferencia es la polarización.\n"
        f"contraste escalar → vectorial por grupo —   {summary}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.92))
    path = out_dir / "resolved_features_layout.png"
    fig.savefig(path, dpi=150)
    print_saved(path)
    plt.close(fig)


def _figure_aerial(dx, shape):
    """Aerial images at the pitches where the two theories disagree."""
    pitches = [1.0, 1.15, 1.5]
    n = shape[0]

    fig, axes = plt.subplots(3, len(pitches), figsize=(4.6 * len(pitches), 11.6))
    for col, f in enumerate(pitches):
        pitch = snap_pitch(f * PITCH_CUT, dx)
        obj = grating(shape, dx, pitch, "vertical")
        span = 3.0 * pitch
        half = max(int(span / dx), 8)
        sl = slice(n // 2 - half, n // 2 + half)
        extent = [-half * dx / lam, half * dx / lam, -half * dx / lam, half * dx / lam]
        axis = (np.arange(n) - n // 2)[sl] * dx / lam

        profiles = {}
        for row, model in enumerate(("scalar", "vectorial")):
            I = total_intensity(image_fields(obj, dx, model=model))
            c = contrast(I, dx, pitch, "vertical")
            profiles[model] = (I[n // 2, sl] / I.max(), c)
            ax = axes[row, col]
            ax.imshow(I[sl, sl] / I.max(), extent=extent, origin="lower",
                      cmap="inferno", vmin=0.0, vmax=1.0)
            ax.set_title(f"{'escalar' if model == 'scalar' else 'vectorial'} — "
                         f"paso {pitch * 1e6:.0f} nm,  C = {c:.3f}", fontsize=10)
            ax.set_xlabel(r"$x/\lambda$")
            if col == 0:
                ax.set_ylabel(r"$y/\lambda$")

        ax = axes[2, col]
        for model, style in (("scalar", "-"), ("vectorial", "-")):
            prof, c = profiles[model]
            ax.plot(axis, prof, style, lw=1.8,
                    label=f"{'escalar' if model == 'scalar' else 'vectorial'} (C={c:.3f})")
        ax.set_xlabel(r"$x/\lambda$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        if col == 0:
            ax.set_ylabel("intensidad normalizada")

    fig.suptitle(
        "Imagen de una rejilla, líneas ⊥ a la polarización (TM), iluminación $\\hat{x}$\n"
        "misma apodización en ambos: la única diferencia es la mezcla de polarización",
        fontsize=11,
    )
    fig.tight_layout()
    path = out_dir / "resolved_features_images.png"
    fig.savefig(path, dpi=150)
    print_saved(path)
    plt.close(fig)


def _figure_mechanism(dx, shape):
    """Where the lost contrast goes: the longitudinal component."""
    pitch = snap_pitch(1.1 * PITCH_CUT, dx)
    n = shape[0]
    half = max(int(2.0 * pitch / dx), 8)
    sl = slice(n // 2 - half, n // 2 + half)
    axis = (np.arange(n) - n // 2)[sl] * dx / lam

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    for ax, orientation in zip(axes[:2], ("vertical", "horizontal")):
        obj = grating(shape, dx, pitch, orientation)
        Ex, Ey, Ez = image_fields(obj, dx, model="vectorial")
        Sx, Sy, Sz = np.abs(Ex) ** 2, np.abs(Ey) ** 2, np.abs(Ez) ** 2
        norm = float((Sx + Sy + Sz).max())
        cut = (lambda A: A[n // 2, sl]) if orientation == "vertical" else (lambda A: A[sl, n // 2])
        ax.plot(axis, cut(Sx) / norm, lw=1.8, label=r"$|E_x|^2$")
        ax.plot(axis, cut(Sy) / norm, lw=1.8, label=r"$|E_y|^2$")
        ax.plot(axis, cut(Sz) / norm, lw=1.8, label=r"$|E_z|^2$")
        ax.plot(axis, cut(Sx + Sy + Sz) / norm, "k--", lw=1.2, label="total")
        meta = ORIENTATIONS[orientation]
        ax.set_title(meta["label"], fontsize=10)
        ax.set_xlabel(r"posición$/\lambda$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("intensidad normalizada")

    ax = axes[2]
    nu = np.linspace(0.0, NU_CUT * 0.9999, 400)
    sin_ai = lam * nu / LENS["ni"]
    r = _radius_from_sine(sin_ai)
    lam_r, _lam_phi, lam_z = transfer_weights_on_grid(r, SURFACE)
    theta = np.arcsin(np.clip(sin_ai, 0.0, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        model_ratio = np.where(lam_r > 0, lam_z / lam_r, 0.0)
    ax.plot(nu / NU_CUT, np.tan(theta), lw=3.2, alpha=0.35, color="k",
            label=r"$\tan\theta$  (proyección $p$ del orden inclinado)")
    ax.plot(nu / NU_CUT, model_ratio, lw=1.8,
            label=r"$\lambda_z/\lambda_r$  (operador de vecdiff)")
    ax.plot(nu / NU_CUT, np.cos(2.0 * theta), lw=1.8,
            label=r"$\cos 2\theta$  (contraste TM de dos haces)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set(xlabel=r"frecuencia espacial $\nu/\nu_{\rm corte}$",
           title="El peso longitudinal del operador es exactamente "
                 r"$\tan\theta$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "El contraste que pierde TM reaparece como componente longitudinal $E_z$",
        fontsize=11,
    )
    fig.tight_layout()
    path = out_dir / "resolved_features_mechanism.png"
    fig.savefig(path, dpi=150)
    print_saved(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
