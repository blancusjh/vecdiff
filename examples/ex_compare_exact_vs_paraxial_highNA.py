import numpy as np
import matplotlib.pyplot as plt
from vecdiff.fresnel import FresnelOvoid, FresnelOvoidParax
from vecdiff.hankel import HankelTransform


# ===================== PARAMETERS =====================
# Physical
LAM = 532e-6
N0 = 1.0
NI = 2.2
Z0 = 2.0
ZI = 8.0
VARPHI = 0.0

# Numerical
R = 2.0
A = 0.95 * R
N = 1000
FOCUS_REFINED = True
FOCUS_POWER = 3.6
PROFILE_POWER = 0  # EL=ER=(r/a)^PROFILE_POWER for r<=a

# Near-focus visualization
Q_VIEW_FRAC = 0.02
Q_ZOOM_FRAC = 0.03
N_IMG = 1300
# ======================================================


def make_grid(max_val, n, refined, power):
    if refined:
        u = np.linspace(0.0, 1.0, n)
        return max_val * u**power
    return np.linspace(0.0, max_val, n)


def sanitize_at_origin(arr, r):
    arr = np.asarray(arr, dtype=float)
    bad = ~np.isfinite(arr)
    if np.any(bad):
        good = ~bad
        arr[bad] = np.interp(r[bad], r[good], arr[good])
    return arr


def radial_map(profile_1d, q, q_view, n_img):
    x = np.linspace(-q_view, q_view, n_img)
    y = np.linspace(-q_view, q_view, n_img)
    X, Y = np.meshgrid(x, y)
    R_img = np.sqrt(X**2 + Y**2)
    img = np.interp(R_img, q, profile_1d, left=profile_1d[0], right=profile_1d[-1])
    extent = [-q_view, q_view, -q_view, q_view]
    return img, extent


def get_fresnel_coeffs(model, r, n0, ni, z0, zi):
    if model == "exact":
        f_exact = FresnelOvoid(n0=n0, z0=z0, ni=ni, zi=zi)
        with np.errstate(divide="ignore", invalid="ignore"):
            ts = sanitize_at_origin(f_exact.ts(r), r)
            tp = sanitize_at_origin(f_exact.tp(r), r)
        return ts, tp

    if model == "paraxial":
        f_parax = FresnelOvoidParax(n0, ni, z0, zi)
        ts, tp = f_parax.coeffs(r)
        return np.asarray(ts, dtype=float), np.asarray(tp, dtype=float)

    raise ValueError(f"Unknown model: {model}")


def transmitted_fields(ts, tp, r, q, EL, ER, varphi):
    ht = HankelTransform.transform_array
    phase = exp(2.0j * varphi)

    e_l = (
        2.0 * π * ht(0, r, (tp + ts) * EL, q)
        - 2.0 * π * phase * ht(1, r, (tp - ts) * ER, q)
    )

    e_r = 2.0 * π * (
        -phase * ht(2, r, (tp - ts) * EL, q)
        + ht(0, r, (tp + ts) * ER, q)
    )

    intensity = np.abs(e_l) ** 2 + np.abs(e_r) ** 2
    return e_l, e_r, intensity


def build_input_fields(r, a, profile_power):
    ap = (r <= a).astype(float)
    profile = (r / a) ** profile_power
    el = profile * ap
    er = profile * ap
    return el, er


def plot_input_fields(r, el, er, a, profile_power):
    input_intensity = np.abs(el) ** 2 + np.abs(er) ** 2
    input_img, extent = radial_map(input_intensity, r, a, 700)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)

    ax[0].plot(r, el, label=r"$E_L(r)$", linewidth=2)
    ax[0].plot(r, er, "--", label=r"$E_R(r)$", linewidth=2)
    ax[0].axvline(a, color="k", linestyle=":", linewidth=1.5, label=r"$r=a$")
    ax[0].set_xlabel(r"$r$")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_title("Input Circular Components")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()
    ax[0].text(
        0.02,
        0.96,
        f"EL=ER=(r/a)^{profile_power} for r<=a, else 0",
        transform=ax[0].transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    im = ax[1].imshow(input_img, extent=extent, origin="lower", cmap="hot")
    im.set_clim(float(np.min(input_img)), float(np.max(input_img)))
    ax[1].set_title(r"Input Scalar Field $|E_L|^2 + |E_R|^2$")
    ax[1].set_xlabel(r"$x$")
    ax[1].set_ylabel(r"$y$")
    ax[1].set_aspect("equal")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)


def plot_1d_zoom(q, q_max, q_zoom_frac, el_ex, el_px, er_ex, er_px, i_ex, i_px, d_el, d_er, d_i):
    q_zoom_max = q_zoom_frac * q_max
    zoom_mask = q <= q_zoom_max

    fig, ax = plt.subplots(2, 3, figsize=(16, 8.2), constrained_layout=True)

    ax[0, 0].plot(q[zoom_mask], np.abs(el_ex)[zoom_mask], label="|E'_L| Exact", linewidth=2)
    ax[0, 0].plot(q[zoom_mask], np.abs(el_px)[zoom_mask], "--", label="|E'_L| Paraxial", linewidth=2)
    ax[0, 0].set_title(r"$E'_L$ near focus")

    ax[0, 1].plot(q[zoom_mask], np.abs(er_ex)[zoom_mask], label="|E'_R| Exact", linewidth=2)
    ax[0, 1].plot(q[zoom_mask], np.abs(er_px)[zoom_mask], "--", label="|E'_R| Paraxial", linewidth=2)
    ax[0, 1].set_title(r"$E'_R$ near focus")

    ax[0, 2].plot(q[zoom_mask], i_ex[zoom_mask], label="I Exact", linewidth=2)
    ax[0, 2].plot(q[zoom_mask], i_px[zoom_mask], "--", label="I Paraxial", linewidth=2)
    ax[0, 2].set_title("Intensity near focus")

    ax[1, 0].plot(q[zoom_mask], d_el[zoom_mask], color="k", linewidth=2, label="Absolute Difference")
    ax[1, 0].set_title(r"$|\,|E'_L|_{ex} - |E'_L|_{px}\,|$")

    ax[1, 1].plot(q[zoom_mask], d_er[zoom_mask], color="k", linewidth=2, label="Absolute Difference")
    ax[1, 1].set_title(r"$|\,|E'_R|_{ex} - |E'_R|_{px}\,|$")

    ax[1, 2].plot(q[zoom_mask], d_i[zoom_mask], color="k", linewidth=2, label="Absolute Difference")
    ax[1, 2].set_title(r"$|I_{ex} - I_{px}|$")

    for axis in ax.flat:
        axis.set_xlabel(r"$q$")
        axis.grid(True, alpha=0.3)
        axis.legend()
    ax[0, 0].set_ylabel("Amplitude")
    ax[0, 1].set_ylabel("Amplitude")
    ax[0, 2].set_ylabel("Intensity")
    ax[1, 0].set_ylabel("Abs. Difference")
    ax[1, 1].set_ylabel("Abs. Difference")
    ax[1, 2].set_ylabel("Abs. Difference")

    fig.suptitle(f"Zoomed Near Focus (q <= {q_zoom_frac:.2f} q_max)", y=1.03)


def plot_2d_maps(q, q_max, q_view_frac, n_img, el_ex, el_px, er_ex, er_px, i_ex, i_px, d_i, summary):
    q_view = q_view_frac * q_max

    el_ex_img, extent = radial_map(np.abs(el_ex), q, q_view, n_img)
    el_px_img, _ = radial_map(np.abs(el_px), q, q_view, n_img)
    er_ex_img, _ = radial_map(np.abs(er_ex), q, q_view, n_img)
    er_px_img, _ = radial_map(np.abs(er_px), q, q_view, n_img)
    i_ex_img, _ = radial_map(i_ex, q, q_view, n_img)
    i_px_img, _ = radial_map(i_px, q, q_view, n_img)
    d_i_img, _ = radial_map(d_i, q, q_view, n_img)

    maps = [
        (el_px_img, r"$|E'_L|$ Paraxial", "hot"),
        (er_px_img, r"$|E'_R|$ Paraxial", "hot"),
        (el_ex_img, r"$|E'_L|$ Exact", "hot"),
        (er_ex_img, r"$|E'_R|$ Exact", "hot"),
        (i_ex_img, "Intensity Exact", "hot"),
        (i_px_img, "Intensity Paraxial", "hot"),
        (d_i_img, "Absolute Intensity Difference |Exact - Parax|", "hot"),
    ]

    fig, ax = plt.subplots(2, 4, figsize=(18, 8.5), constrained_layout=True)

    for i, (img, title, cmap) in enumerate(maps):
        rr = i // 4
        cc = i % 4
        im = ax[rr, cc].imshow(img, extent=extent, origin="lower", cmap=cmap)
        im.set_clim(float(np.min(img)), float(np.max(img)))
        ax[rr, cc].set_title(title)
        ax[rr, cc].set_xlabel(r"$x$")
        ax[rr, cc].set_ylabel(r"$y$")
        ax[rr, cc].set_aspect("equal")
        fig.colorbar(im, ax=ax[rr, cc], fraction=0.046, pad=0.04)

    ax[1, 3].axis("off")
    ax[1, 3].text(0.02, 0.95, summary, va="top")

    fig.suptitle("2D Scalar Comparison: Exact vs Paraxial", y=1.02)


def main():
    q_max = NI / (LAM * ZI) * R
    r = make_grid(R, N, FOCUS_REFINED, FOCUS_POWER)
    q = make_grid(q_max, N, FOCUS_REFINED, FOCUS_POWER)

    el, er = build_input_fields(r, A, PROFILE_POWER)

    ts_exact, tp_exact = get_fresnel_coeffs("exact", r, N0, NI, Z0, ZI)
    ts_parax, tp_parax = get_fresnel_coeffs("paraxial", r, N0, NI, Z0, ZI)

    el_exact, er_exact, i_exact = transmitted_fields(ts_exact, tp_exact, r, q, el, er, VARPHI)
    el_parax, er_parax, i_parax = transmitted_fields(ts_parax, tp_parax, r, q, el, er, VARPHI)

    d_el = np.abs(np.abs(el_exact) - np.abs(el_parax))
    d_er = np.abs(np.abs(er_exact) - np.abs(er_parax))
    d_i = np.abs(i_exact - i_parax)

    rel_i_diff = np.trapezoid(d_i * q, q) / (np.trapezoid(i_exact * q, q) + 1e-15)
    print(f"Relative intensity difference (Exact vs Paraxial): {rel_i_diff:.5f}")

    summary = (
        "Summary\n"
        f"Relative |ΔI|/|I_exact| = {rel_i_diff:.5f}\n"
        f"n0={N0}, ni={NI}, z0={Z0}, zi={ZI}\n"
        f"R={R}, a/R={A/R:.2f}, N={N}\n"
        f"Focus refined={FOCUS_REFINED}, power={FOCUS_POWER}\n"
        f"q-view={Q_VIEW_FRAC:.2f} q_max, q-zoom={Q_ZOOM_FRAC:.2f} q_max\n"
        f"EL=ER=(r/a)^{PROFILE_POWER} inside aperture"
    )

    plot_input_fields(r, el, er, A, PROFILE_POWER)
    plot_1d_zoom(
        q,
        q_max,
        Q_ZOOM_FRAC,
        el_exact,
        el_parax,
        er_exact,
        er_parax,
        i_exact,
        i_parax,
        d_el,
        d_er,
        d_i,
    )
    plot_2d_maps(
        q,
        q_max,
        Q_VIEW_FRAC,
        N_IMG,
        el_exact,
        el_parax,
        er_exact,
        er_parax,
        i_exact,
        i_parax,
        d_i,
        summary,
    )

    plt.show()


if __name__ == "__main__":
    main()
