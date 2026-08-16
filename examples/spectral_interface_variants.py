"""Which incidence cosine: one per surface point, or one per plane wave?

The direct route from medium 1 to medium 2 is: evaluate the field on the
interface, split it with Fresnel, carry the transmitted field back to a tangent
plane.  The only place a choice has to be made is where ``cos(theta_i)`` comes
from --

* ``local``     one direction per surface point, read from the phase gradient
                of the field along the surface;
* ``spectral``  one direction per plane wave, ``k_hat . n_hat``, summed over the
                whole angular spectrum.

This script decides between them, in the meridional plane where the vector
problem splits exactly into the TE and TM scalar problems, so nothing is lost
by working with two scalars.  Five measurements:

1.  **Plane interface.**  There the tangent-plane model is not a model at all --
    it is the rigorous solution -- so ``spectral`` is exact by construction and
    every discrepancy is charged to ``local``.  Sweeping the angular spread of
    the illumination turns the comparison into a law rather than an anecdote.

2.  **Curved interface, one source.**  One ray reaches each surface point, so
    ``local`` is exact in principle.  Measured with a soft pupil edge and with
    a hard one, because a hard edge radiates its own wave and that second wave
    is what the single-direction model has no way to carry.

3.  **Curved interface, two sources.**  Two rays reach each point.  The true
    operator is linear in the incident field; ``spectral`` is linear to machine
    precision, ``local`` is not.  The linearity defect is measured directly,
    which settles the case without appeal to any external reference.

4.  **The return trip.**  Pulling the surface field back to a tangent plane with
    ``exp(-i kz z(Sigma))`` is the same kappa-by-Q operator read backwards, not
    a pointwise multiplication.  Both readings are measured.

5.  **Cost.**  Both kernels depend on the pair ``(kappa, Q)`` through a smooth
    function of a bilinear form, so both are numerically low rank: the sum over
    the spectrum collapses to a handful of transforms.

Run from the repository root::

    uv run python examples/spectral_interface_variants.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vecdiff import CartesianSurface  # noqa: E402
from vecdiff.spectral_interface import (  # noqa: E402
    PlaneWaveSet,
    flat_profile,
    fresnel_operator_rank,
    line_source,
    local_incidence_sine,
    profile_from_surface,
    radiate,
    surface_field_local,
    surface_field_spectral,
    transmission,
)

π = np.pi

OUTPUT = Path(__file__).resolve().parent / "output"

LAMBDA = 1.0                 # everything in wavelengths of the incident medium
N1, N2 = 1.0, 1.5
K1 = 2.0 * π * N1 / LAMBDA
K2 = 2.0 * π * N2 / LAMBDA

SURFACE = CartesianSurface(n0=N1, ni=N2, z0=-40.0, zi=24.0)

#: Stay clear of the rim.  The oval reaches grazing incidence at its aperture
#: limit, where the normal is nearly perpendicular to the ray; past ~0.7 of it
#: an off-axis plane wave of the band below would strike the surface from
#: behind, which is outside the tangent-plane model for either channel.
APERTURE = 0.70 * SURFACE.aperture_limit
SIN_MAX = 0.45               # half-width of the sampled angular band
SOURCE_OFFSET = 11.0         # lateral offset of the two sources, wavelengths


def rel_error(a, b, weight=None):
    """Relative L2 distance between two fields, optionally intensity-weighted."""
    if weight is None:
        weight = np.ones_like(np.abs(b))
    num = np.sum(weight * np.abs(a - b) ** 2)
    den = np.sum(weight * np.abs(b) ** 2)
    return float(np.sqrt(num / den))


# ------------------------------------------------------------------ #
#  1. Plane interface: the law of the local model                     #
# ------------------------------------------------------------------ #

def test_plane(spreads_deg, mean_deg=35.0, n_x=4001, half_width=40.0):
    """Two plane waves separated by ``spread``; error of each channel vs truth.

    On a plane the rigorous transmitted field is Fresnel applied to each plane
    wave separately -- exactly what ``spectral`` computes.  ``local`` has to
    invent one direction for the pair, and the first-order prediction for its
    error is ``|d ln t / d theta| * spread / 2``.
    """
    x = np.linspace(-half_width, half_width, n_x)
    profile = flat_profile(x)
    rows = []
    for spread in spreads_deg:
        th = np.radians(mean_deg) + np.array([-0.5, 0.5]) * np.radians(spread)
        pw = PlaneWaveSet(kx=K1 * np.sin(th), amp=np.ones(2, dtype=complex), k=K1)
        row = {"spread": spread}
        for pol in ("TE", "TM"):
            t, _ = transmission(np.cos(th), N1, N2, pol)
            exact = (pw.phasor(profile.x, profile.z) * t * pw.amp).sum(axis=1)
            psi_s, _ = surface_field_spectral(pw, profile, N1, N2, pol)
            psi_l, _, sin_i = surface_field_local(pw, profile, N1, N2, pol)
            row[f"spectral_{pol}"] = rel_error(psi_s, exact)
            row[f"local_{pol}"] = rel_error(psi_l, exact)
            row["superosc"] = float(np.mean(np.abs(sin_i) > 1.0))
        rows.append(row)
    return rows


def predicted_local_slope(mean_deg, polarization, delta=1e-4):
    """``|d ln t / d theta| / 2``, the first-order coefficient of the law above."""
    th = np.radians(mean_deg)
    t_p, _ = transmission(np.cos(th + delta), N1, N2, polarization)
    t_m, _ = transmission(np.cos(th - delta), N1, N2, polarization)
    t_0, _ = transmission(np.cos(th), N1, N2, polarization)
    return float(abs((t_p - t_m) / (2.0 * delta) / t_0) / 2.0)


# ------------------------------------------------------------------ #
#  2-3. Curved interface                                              #
# ------------------------------------------------------------------ #

def build_pupil(n_x=3201):
    return profile_from_surface(SURFACE, np.linspace(-APERTURE, APERTURE, n_x))


def build_spectrum(sources, n_k=1201, *, soft_edge=True):
    """Angular spectrum of a set of line sources, band-limited to the pupil.

    ``soft_edge`` tapers the band instead of cutting it.  A hard cut is a stop
    in the far field and it radiates an edge wave; keeping the two cases apart
    is what separates the model's own error from the aperture's.
    """
    kx = np.linspace(-SIN_MAX, SIN_MAX, n_k) * K1
    amp = np.zeros(n_k, dtype=complex)
    for x_s, z_s, a in sources:
        amp += line_source(kx, K1, x_s, z_s, amplitude=a)
    if soft_edge:
        amp = amp * np.exp(-4.0 * (kx / (SIN_MAX * K1)) ** 8)
    return PlaneWaveSet(kx=kx, amp=amp * (kx[1] - kx[0]), k=K1)


def surface_models(pw, profile, pol):
    psi_s, dpsi_s = surface_field_spectral(pw, profile, N1, N2, pol)
    psi_l, dpsi_l, sin_i = surface_field_local(pw, profile, N1, N2, pol)
    return (psi_s, dpsi_s), (psi_l, dpsi_l), sin_i


def image_cut(profile, boundary, x_obs, z_obs):
    obs = np.stack([x_obs, np.full_like(x_obs, z_obs)], axis=-1)
    return radiate(profile, boundary[0], boundary[1], obs, K2)


# ------------------------------------------------------------------ #
#  4. The return trip to the tangent plane                            #
# ------------------------------------------------------------------ #

def gaussian_beam(k, centres, *, width=0.10, band=0.60, n_k=1201):
    """A sum of Gaussian lobes in direction, confined in ``x``.

    One lobe is a beam with a single dominant ``kz``; two well-separated lobes
    put a wide range of ``kz`` at every point of the beam while keeping the
    field inside the window, which is what separates the two readings of the
    pullback without letting the aperture contaminate the measurement.
    """
    kx = np.linspace(-band, band, n_k) * k
    amp = np.zeros(n_k)
    for sin0 in centres:
        amp = amp + np.exp(-((kx / k - sin0) / width) ** 2)
    return PlaneWaveSet(kx=kx, amp=amp.astype(complex) * (kx[1] - kx[0]), k=k)


def test_pullback(pw, profile, *, sag_scale=1.0):
    """Carry a beam to the surface and back to the tangent plane, three ways.

    The comparison is made in real space against the field the same spectrum
    has on ``z = 0``, because a spectrum reconstructed from a finite window is
    smoothed by the window and that has nothing to do with the operator.

    ``adjoint``
        The recipe read literally: multiply by ``exp(-i kz z(Sigma))`` and
        transform.  This is the *adjoint* of the outward map.
    ``pointwise``
        The same with one ``kz`` for the whole beam, so no transform at all.
    ``solve``
        The actual inverse: find the spectrum whose surface values are the ones
        observed, by least squares on the same matrix.  Its residual says
        whether an exact inverse exists at all -- and it does, which makes the
        adjoint's error a choice rather than a limitation.
    """
    z = sag_scale * profile.z
    truth = pw.field(profile.x, np.zeros_like(profile.x))
    forward = np.exp(1j * (pw.kx[None, :] * profile.x[:, None]
                           + pw.kz[None, :] * z[:, None]))
    on_surface = forward @ pw.amp
    dx = float(np.mean(np.diff(profile.x)))
    dkx = float(np.mean(np.diff(pw.kx)))
    to_plane = np.exp(1j * pw.kx[None, :] * profile.x[:, None])

    adjoint = to_plane @ ((forward.conj().T @ on_surface) * dx / (2.0 * π) * dkx)
    kz_ref = pw.kz[np.argmax(np.abs(pw.amp))]
    pointwise = on_surface * np.exp(-1j * kz_ref * z)
    solved, *_ = np.linalg.lstsq(forward, on_surface, rcond=1e-8)

    def fit(candidate):
        scale = np.vdot(truth, candidate) / np.vdot(truth, truth)
        return rel_error(candidate / scale, truth)

    return fit(adjoint), fit(pointwise), fit(to_plane @ solved)


# ------------------------------------------------------------------ #
#  Driver                                                             #
# ------------------------------------------------------------------ #

def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT.mkdir(parents=True, exist_ok=True)
    edge = SURFACE.ray_geometry(np.array([APERTURE]))
    print(f"system: n1={N1}, n2={N2}, z0={SURFACE.z0}, zi={SURFACE.zi} "
          f"(wavelengths)")
    print(f"pupil r <= {APERTURE:.2f} = {APERTURE/SURFACE.aperture_limit:.2f} "
          f"of the grazing limit; image-side NA at the rim "
          f"= {N2*edge.sin_ai[0]:.3f}\n")

    # -- 1 ------------------------------------------------------------ #
    spreads = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0])
    rows = test_plane(spreads)
    print("1. PLANE INTERFACE  (the tangent-plane model is exact there)")
    print(f"{'spread':>7} {'spectral TE':>12} {'spectral TM':>12} "
          f"{'local TE':>10} {'local TM':>10}")
    print("-" * 56)
    for row in rows:
        print(f"{row['spread']:>6.0f}d {row['spectral_TE']:>12.2e} "
              f"{row['spectral_TM']:>12.2e} {row['local_TE']:>10.2e} "
              f"{row['local_TM']:>10.2e}")
    for pol in ("TE", "TM"):
        slope = predicted_local_slope(35.0, pol)
        measured = rows[1][f"local_{pol}"] / np.radians(1.0)
        print(f"   {pol}: |d ln t/d theta|/2 = {slope:.3f} /rad, "
              f"measured slope at 1 deg = {measured:.3f} /rad")
    print()

    # -- 2 ------------------------------------------------------------ #
    profile = build_pupil()
    print("2. CURVED INTERFACE, ONE SOURCE  (one ray reaches each point)")
    single, single_res = {}, {}
    for tag, soft in (("borde suave", True), ("borde duro", False)):
        pw = build_spectrum([(0.0, SURFACE.z0, 1.0)], soft_edge=soft)
        single[tag] = pw
        for pol in ("TE", "TM"):
            (psi_s, d_s), (psi_l, d_l), sin_i = surface_models(pw, profile, pol)
            err = rel_error(psi_l, psi_s, weight=np.abs(psi_s) ** 2)
            single_res[(tag, pol)] = (psi_s, d_s, psi_l, d_l, sin_i, err)
            print(f"   {tag:<12} {pol}: local vs spectral = {err:.3e}"
                  f"   (|sin|>1 on {np.mean(np.abs(sin_i) > 1.0):.1%} of the pupil)")
    print()

    # -- 3 ------------------------------------------------------------ #
    d = SOURCE_OFFSET
    pair = build_spectrum([(-d, SURFACE.z0, 1.0), (d, SURFACE.z0, 1.0)])
    src_a = build_spectrum([(-d, SURFACE.z0, 1.0)])
    src_b = build_spectrum([(d, SURFACE.z0, 1.0)])
    print(f"3. CURVED INTERFACE, TWO SOURCES at x = +-{d:.0f}  "
          f"(two rays reach each point)")
    pair_res = {}
    for pol in ("TE", "TM"):
        (psi_s, d_s), (psi_l, d_l), sin_i = surface_models(pair, profile, pol)
        psi_a, _ = surface_field_spectral(src_a, profile, N1, N2, pol)
        psi_b, _ = surface_field_spectral(src_b, profile, N1, N2, pol)
        la, _, _ = surface_field_local(src_a, profile, N1, N2, pol)
        lb, _, _ = surface_field_local(src_b, profile, N1, N2, pol)
        err = rel_error(psi_l, psi_s, weight=np.abs(psi_s) ** 2)
        pair_res[pol] = (psi_s, d_s, psi_l, d_l, sin_i, err)
        print(f"   {pol}: defecto de linealidad  spectral = "
              f"{rel_error(psi_s, psi_a + psi_b):.2e}   "
              f"local = {rel_error(psi_l, la + lb):.2e}")
        print(f"       local vs spectral = {err:.3e}"
              f"   (|sin|>1 on {np.mean(np.abs(sin_i) > 1.0):.1%} of the pupil)")
    print()

    x_obs = np.linspace(-12.0, 12.0, 601)
    images = {}
    for pol in ("TE", "TM"):
        psi_s, d_s, psi_l, d_l, _, _ = single_res[("borde suave", pol)]
        images[("una fuente", pol)] = (
            np.abs(image_cut(profile, (psi_s, d_s), x_obs, SURFACE.zi)) ** 2,
            np.abs(image_cut(profile, (psi_l, d_l), x_obs, SURFACE.zi)) ** 2,
        )
        psi_s, d_s, psi_l, d_l, _, _ = pair_res[pol]
        images[("dos fuentes", pol)] = (
            np.abs(image_cut(profile, (psi_s, d_s), x_obs, SURFACE.zi)) ** 2,
            np.abs(image_cut(profile, (psi_l, d_l), x_obs, SURFACE.zi)) ** 2,
        )
    print("   plano imagen, diferencia L2 relativa local vs spectral")
    for key, (Is, Il) in images.items():
        print(f"      {key[0]:<12} {key[1]}: "
              f"{np.sqrt(np.sum((Il-Is)**2)/np.sum(Is**2)):.3e}")
    print()

    # -- 4 ------------------------------------------------------------ #
    print("4. RETURN TRIP  exp(-i kz z(Sigma))")
    # A wider window than the pupil, so the beam is not clipped: the round trip
    # is a statement about the operator, not about the aperture.
    x_wide = np.linspace(-0.95 * SURFACE.aperture_limit,
                         0.95 * SURFACE.aperture_limit, 1501)
    curved_wide = profile_from_surface(SURFACE, x_wide)
    beam = gaussian_beam(K2, [-0.30, 0.30])
    print(f"{'sagita':>10} {'k*dz':>8} {'adjunto':>11} {'puntual':>11} {'inverso':>11}")
    print("-" * 56)
    pull = []
    for scale in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
        e_adj, e_pt, e_inv = test_pullback(beam, curved_wide, sag_scale=scale)
        k_dz = K2 * scale * (curved_wide.z.max() - curved_wide.z.min())
        pull.append((k_dz, e_adj, e_pt, e_inv))
        label = "plana" if scale == 0.0 else f"x{scale:g}"
        print(f"{label:>10} {k_dz:>8.1f} {e_adj:>11.2e} {e_pt:>11.2e} {e_inv:>11.2e}")
    print()

    # -- 5 ------------------------------------------------------------ #
    print("5. COST")
    for pol in ("TE", "TM"):
        rank10, sv = fresnel_operator_rank(pair, profile, N1, N2, pol)
        print(f"   {pol}: Fresnel kernel {profile.x.size} x {pair.kx.size}, "
              f"rango numérico {int(np.sum(sv > 1e-6*sv[0]))} a 1e-6, "
              f"{rank10} a 1e-10")
    sag_kernel = np.exp(1j * pair.kz[None, :] * profile.z[:, None])
    sv_sag = np.linalg.svd(sag_kernel, compute_uv=False)
    k_dz = K1 * (profile.z.max() - profile.z.min())
    print(f"   sagita exp(i kz z): rango {int(np.sum(sv_sag > 1e-6*sv_sag[0]))} "
          f"a 1e-6 para k*dz = {k_dz:.0f}")
    depths = []
    for frac in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        kernel = np.exp(1j * pair.kz[None, :] * (frac * profile.z[:, None]))
        s = np.linalg.svd(kernel, compute_uv=False)
        depths.append((frac * k_dz, int(np.sum(s > 1e-6 * s[0]))))
    print("   rango de la sagita frente a la profundidad:  "
          + ",  ".join(f"{a:.0f}->{b}" for a, b in depths)
          + "   (k*dz -> rango)")
    print()

    # ---------------------------------------------------------------- #
    #  figure                                                           #
    # ---------------------------------------------------------------- #
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.2))

    ax = axes[0, 0]
    for pol, c in (("TE", "tab:blue"), ("TM", "tab:red")):
        ax.loglog(spreads[1:], [r[f"local_{pol}"] for r in rows[1:]],
                  "o-", color=c, label=f"local, {pol}")
        ax.loglog(spreads[1:], [max(r[f"spectral_{pol}"], 1e-17) for r in rows[1:]],
                  "s--", color=c, alpha=0.4, label=f"spectral, {pol}")
    ref = predicted_local_slope(35.0, "TE") * np.radians(spreads[1:])
    ax.loglog(spreads[1:], ref, "k:", lw=1.2,
              label=r"$\frac{1}{2}|\mathrm{d}\ln t/\mathrm{d}\theta|\,\Delta\theta$")
    ax.set_xlabel(r"separación angular de las dos ondas $\Delta\theta$ [grados]")
    ax.set_ylabel("error relativo en la interfaz")
    ax.set_title("1. Interfaz plana: la respuesta rigurosa se conoce")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[0, 1]
    sin_soft = local_incidence_sine(
        single["borde suave"].field(profile.x, profile.z), profile, K1)
    sin_pair = local_incidence_sine(pair.field(profile.x, profile.z), profile, K1)
    ax.plot(profile.x, sin_soft, lw=1.2, label="una fuente")
    ax.plot(profile.x, sin_pair, lw=0.7, label="dos fuentes")
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.axhline(-1.0, color="k", ls="--", lw=0.8)
    ax.set_ylim(-3.0, 3.0)
    ax.set_xlabel(r"$x$ sobre la interfaz  [$\lambda$]")
    ax.set_ylabel(r"$\sin\theta_i$ del gradiente de fase")
    ax.set_title("2-3. La dirección local, cuando existe y cuando no")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    k_dz = np.array([p[0] for p in pull])
    ax.semilogy(k_dz, [p[1] for p in pull], "o-", label=r"adjunto  $e^{-ik_zz(\Sigma)}$")
    ax.semilogy(k_dz, [p[2] for p in pull], "s-", label=r"puntual  $k_z$ de referencia")
    ax.semilogy(k_dz, [max(p[3], 1e-16) for p in pull], "^-", label="inverso exacto")
    ax.semilogy(k_dz, 7e-4 * k_dz, "k:", lw=1.2, label=r"$7\times10^{-4}\,k\,\Delta z$")
    ax.set_xlabel(r"profundidad de sagita  $k\,\Delta z$  [rad]")
    ax.set_ylabel("error del viaje de vuelta")
    ax.set_title("4. Volver al plano tangente")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    for col, tag in enumerate(("una fuente", "dos fuentes")):
        ax = axes[1, col]
        Is, Il = images[(tag, "TM")]
        ax.plot(x_obs, Is / Is.max(), "k", lw=2.6, alpha=0.4, label="spectral")
        ax.plot(x_obs, Il / Is.max(), "tab:red", lw=1.1, label="local")
        ax.set_xlabel(r"$x$ en el plano imagen  [$\lambda$]")
        ax.set_ylabel("intensidad normalizada")
        ax.set_title(f"Imagen, TM — {tag}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    ax = axes[1, 2]
    _, sv_f = fresnel_operator_rank(pair, profile, N1, N2, "TM")
    ax.semilogy(np.arange(1, 41), sv_f[:40] / sv_f[0], "o-", ms=3,
                label=r"Fresnel  $t(\hat k\cdot\hat n)$")
    ax.semilogy(np.arange(1, 41), sv_sag[:40] / sv_sag[0], "s-", ms=3,
                label=r"sagita  $e^{ik_z z}$")
    ax.axhline(1e-6, color="k", ls="--", lw=0.8)
    ax.set_xlabel("índice del valor singular")
    ax.set_ylabel("valor singular normalizado")
    ax.set_title("5. Coste: rango numérico de cada factor")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Coseno de incidencia: uno por punto (local) frente a uno por onda plana "
        f"(spectral) — $n_1$={N1}, $n_2$={N2}, NA$_i$={N2*edge.sin_ai[0]:.2f}",
        fontsize=12,
    )
    fig.tight_layout()
    path = OUTPUT / "spectral_interface_variants.png"
    fig.savefig(path, dpi=135)
    print(f"figura: {path}")
    return path


if __name__ == "__main__":
    main()
