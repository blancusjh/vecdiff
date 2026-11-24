"""Figures of the Brewster devices: geometry, needle fields, profiles, cross-check.

Run after run_brewster_axicon.py and run_shell_stack.py with the same output
directory (default: figures/ next to this file). Needs matplotlib.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.special import j0, j1  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import devices as dv  # noqa: E402
import meridional_operator as mo  # noqa: E402
import run_brewster_axicon as ax  # noqa: E402

BLUE = LinearSegmentedColormap.from_list(
    "blue", ["#ffffff", "#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
)
ORANGE = LinearSegmentedColormap.from_list(
    "orange", ["#ffffff", "#fbd9c8", "#f39a72", "#eb6834", "#b4471c", "#6e2a0f"]
)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#222222", "#6b6b66"
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 200,
    }
)
LAM = ax.WAVELENGTH


def trace_axicon(axis, n, depth, label):
    """Draw one double-Brewster axicon and a fan of rays through it (µm)."""
    b = dv.brewster(n)
    surfaces, (inner, outer) = dv.brewster_axicon(0.0, n, depth)
    g2, d = b["exit_slope"], b["delta"]
    r = np.linspace(0, outer, 200)
    zs = np.r_[n * r, (depth + g2 * r)[::-1], depth + g2 * r, (n * r)[::-1]]
    rs = np.r_[r, r[::-1], -r, -r[::-1]]
    axis.fill(zs * 1e6, rs * 1e6, color="#cde2fb", ec=C1, lw=1)
    for rho in np.linspace(inner * 1.05, outer * 0.97, 7):
        z1 = n * rho
        t = (depth + g2 * rho - z1) / (np.cos(d) + g2 * np.sin(d))
        rho2, z2 = rho - t * np.sin(d), z1 + t * np.cos(d)
        cross = z2 + rho2 / np.tan(2 * d)
        for sign in (1, -1):
            axis.plot(
                np.r_[-60e-6, z1, z2, cross + 0.3 * (cross - z2)] * 1e6,
                sign * np.r_[rho, rho, rho2, -0.3 * rho2] * 1e6,
                color=C2,
                lw=0.9,
            )
    axis.set_aspect("equal")
    axis.set_xlabel("z (µm)")
    axis.set_ylabel("ρ (µm), meridional section")
    axis.set_title(
        f"{label}\nθ_B = {np.degrees(b['theta_b']):.2f}°, δ = {np.degrees(d):.2f}°, "
        f"output half-angle 2δ = {np.degrees(2 * d):.1f}° (NA {b['na']:.3f})",
        color=INK,
        fontsize=8.5,
    )


def figure_geometry(out):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    trace_axicon(axs[0], np.sqrt(3), ax.DEPTH, "n = √3, flat exit face")
    trace_axicon(axs[1], 2.0, ax.DEPTH, "n = 2, exit cone of slope 0.182")
    fig.tight_layout()
    fig.savefig(out / "fig_axicon_geometry.png")
    plt.close(fig)


def figure_needle(out, grid):
    cases = (
        ("Brewster axicon + 4 Brewster shells, n = 2", {}),
        ("same geometry, no diattenuation (t_s = t_p)", {"diattenuation": False}),
    )
    rho = np.linspace(-1.5 * LAM, 1.5 * LAM, 241)
    fig, axs = plt.subplots(2, 4, figsize=(12, 5.6), layout="constrained")
    for row, (title, kwargs) in enumerate(cases):
        result = ax.case(grid, 2.0, 4, **kwargs)
        m = ax.metrics(grid, result)
        dz = np.linspace(0, 2.2 * m["peak_dz_um"] * 1e-6, 241)
        maps = ax.needle(grid, result, np.abs(rho), dz)
        total = maps.sum(0)
        peak = total.max()
        extent = [rho[0] / LAM, rho[-1] / LAM, dz[0] * 1e6, dz[-1] * 1e6]
        names = ("|E_ρ|²", "|E_φ|²", "|E_z|²", "|E|²")
        for col, (image, name) in enumerate(zip(list(maps) + [total], names)):
            im = axs[row, col].imshow(
                image / peak,
                extent=extent,
                origin="lower",
                aspect="auto",
                cmap=ORANGE if col == 1 else BLUE,
                vmin=0,
                vmax=1,
            )
            axs[row, col].set_title(f"{name} / max |E|²", color=INK, fontsize=8.5)
            axs[row, col].set_xlabel("ρ (wavelengths)")
        axs[row, 0].set_ylabel(f"{title}\nz behind the exit vertex (µm)", fontsize=8)
    fig.colorbar(im, ax=axs, fraction=0.02, label="intensity / peak total intensity")
    fig.savefig(out / "fig_axicon_needle.png")
    plt.close(fig)


def total_fwhm(na, te_share):
    """Closed-form FWHM (wavelengths) of an ideal conical wave of half-angle
    arcsin(na): TM part sin² J0² + cos² J1², TE part J1², TE/TM power te_share."""
    s2 = na**2

    def intensity(x):
        return s2 * j0(x) ** 2 + (1 - s2) * j1(x) ** 2 + te_share * j1(x) ** 2

    x = brentq(lambda x: intensity(x) - intensity(0) / 2, 1e-6, 2.4)
    return 2 * x / (2 * np.pi * na)


def figure_profiles(out):
    data = np.load(out / "axicon_profiles.npz")
    table = json.loads((out / "axicon_results.json").read_text())
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for kind, col, style in (
        ("brewster", C1, "-"),
        ("no diattenuation", C2, "-"),
        ("radial input", INK, "--"),
    ):
        key = f"n2+shells|{kind}|"
        rho = data[key + "rho"]
        total = data[key + "rho2"] + data[key + "phi2"] + data[key + "z2"]
        label = {
            "brewster": "Brewster axicon + 4 shells",
            "no diattenuation": "same geometry, t_s = t_p",
            "radial input": "ideal radial input",
        }[kind]
        fw = table[f"n2+shells | {kind}"]["fwhm_total_wavelengths"]
        axs[0].plot(rho / LAM, total / total[0], color=col, ls=style, lw=1.6,
                    label=f"{label} (FWHM {fw:.3f} λ)")
    axs[0].set_xlabel("ρ (wavelengths)")
    axs[0].set_ylabel("|E|² / on-axis |E|²")
    axs[0].set_title("needle cross-section at its brightest plane, n = 2 (NA 0.96)", color=INK, fontsize=8.5)
    axs[0].legend(frameon=False, fontsize=7.5)
    key = "n2+shells|brewster|"
    rho = data[key + "rho"]
    peak = (data[key + "rho2"] + data[key + "phi2"] + data[key + "z2"])[0]
    for name, col, lab in (("z2", C1, "|E_z|²"), ("rho2", C3, "|E_ρ|²"), ("phi2", C2, "|E_φ|²")):
        axs[1].plot(rho / LAM, data[key + name] / peak, color=col, lw=1.6, label=lab)
    axs[1].set_xlabel("ρ (wavelengths)")
    axs[1].set_ylabel("component / on-axis |E|²")
    axs[1].set_title("components of the Brewster needle (axicon + 4 shells)", color=INK, fontsize=8.5)
    axs[1].legend(frameon=False, fontsize=8)
    n = np.linspace(1.3, 3.0, 400)
    b = dv.brewster(n)
    axs[2].plot(n, b["na"], color=C1, lw=2, label="NA = sin 2δ = 4n(n²-1)/(n²+1)²")
    axs[2].plot(n, b["power_ratio"], color=C2, lw=2, label="c = (t_s/t_p)² per face = cos² δ")
    for value, name in ((np.sqrt(3), "√3"), (2.0, "2"), (1 + np.sqrt(2), "1+√2")):
        bb = dv.brewster(value)
        axs[2].plot([value], [bb["na"]], "o", color=C1, ms=5)
        axs[2].annotate(name, (value, bb["na"]), xytext=(4, -12), textcoords="offset points", color=INK, fontsize=8)
    axs[2].set_xlabel("refractive index n")
    axs[2].set_ylabel("value")
    axs[2].set_title("double-Brewster axicon, closed forms", color=INK, fontsize=8.5)
    axs[2].legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0.06, 0.0))
    fig.tight_layout()
    fig.savefig(out / "fig_axicon_profiles.png")
    plt.close(fig)
    return {
        "closed-form FWHM, ideal TM, NA 0.96": total_fwhm(0.96, 0.0),
        "closed-form FWHM, TE = TM, NA 0.96": total_fwhm(0.96, 1.0),
        "closed-form FWHM, TE/TM = 0.64^10, NA 0.96": total_fwhm(0.96, 0.64**10),
        "closed-form FWHM, ideal TM, NA 0.866": total_fwhm(np.sqrt(3) / 2, 0.0),
    }


def figure_shells(out):
    data = np.load(out / "shell_fields.npz")
    results = json.loads((out / "shell_results.json").read_text())
    rho = data["rho"] * 1e6
    m = rho < 130
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    labels = {
        "ABABAB|2.0|1e-05|0.0": ("alternating ABABAB, sharp tips", C1),
        "AAAAAA|2.0|1e-05|0.0": ("all spikes AAAAAA", C2),
        "ABABAB|2.0|3e-05|0.0": ("alternating, walls 13.4 µm", C3),
        "ABABAB|2.0|1e-05|5e-06": ("alternating, tips rounded 5 µm", C4),
    }
    ideal = data["ABABAB|2.0|1e-05|0.0|ideal_rho"]
    norm = np.max(np.abs(ideal) ** 2)
    for key, (label, col) in labels.items():
        axs[0].plot(rho[m], np.abs(data[key + "|out_rho"][m]) ** 2 / norm, color=col, lw=1.5, label=label)
    axs[0].plot(rho[m], np.abs(ideal[m]) ** 2 / norm, color=INK, ls="--", lw=1, label="ideal diag(1, c^(N/2))")
    axs[0].set_xlabel("ρ (µm)")
    axs[0].set_ylabel("|E_ρ|² / max ideal |E_ρ|²")
    axs[0].set_title("output radial profile, 1 mm after six shells (n = 2)", color=INK, fontsize=8.5)
    axs[0].legend(frameon=False, fontsize=7.5)
    base = results[0]
    k = [s["shell"] for s in base["shells"]]
    axs[1].semilogy(k, [s["azimuthal_share"] for s in base["shells"]], "o", color=C1, ms=6, label="operator, n = 2")
    kk = np.linspace(1, 6, 200)
    c = 0.64
    axs[1].semilogy(kk, c ** (2 * kk) / (1 + c ** (2 * kk)), color=MUTED, lw=1, label="c^N/(1+c^N), N = 2 × shells")
    bk = results[3]
    axs[1].semilogy([s["shell"] for s in bk["shells"]], [s["azimuthal_share"] for s in bk["shells"]], "s", color=C2, ms=5, label="operator, BK7")
    cb = dv.brewster(1.515)["power_ratio"]
    kk = np.linspace(1, 14, 200)
    axs[1].semilogy(kk, cb ** (2 * kk) / (1 + cb ** (2 * kk)), color=MUTED, lw=1, ls=":")
    axs[1].set_xlabel("shells (integer)")
    axs[1].set_ylabel("E_φ share of the output power")
    axs[1].set_title("suppression per shell: markers simulated, lines closed form", color=INK, fontsize=8.5)
    axs[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "fig_shell_stack.png")
    plt.close(fig)


def figure_crosscheck(out):
    import cross_validate as cv

    lib, own, radii = cv.main()
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.3))
    r = radii * 1e6
    for k, (name, col) in enumerate((("E_ρ", C1), ("E_φ", C2))):
        norm = np.max(np.abs(lib[0]) ** 2)
        axs[0].plot(r, np.abs(lib[k]) ** 2 / norm, color=col, lw=2.2, label=f"|{name}|², vecdiff Propagator")
        axs[0].plot(r, np.abs(own[k]) ** 2 / norm, color=INK, lw=0.9, ls="--", label=f"|{name}|², meridional operator")
        axs[1].plot(r, np.unwrap(np.angle(lib[k] * np.conj(own[k]))), color=col, lw=1.5, label=name)
    axs[0].set_xlabel("ρ (µm)")
    axs[0].set_ylabel("intensity / max |E_ρ|²")
    axs[0].set_title("one shell, tips a = 20 µm, 1 mm behind: intensities", color=INK, fontsize=8.5)
    axs[0].legend(frameon=False, fontsize=7.5)
    axs[1].set_xlabel("ρ (µm)")
    axs[1].set_ylabel("phase difference (rad)")
    axs[1].set_title("phase of vecdiff minus phase of the operator", color=INK, fontsize=8.5)
    axs[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "fig_crosscheck.png")
    plt.close(fig)


def main(out=None):
    out = Path(out) if out else HERE / "figures"
    grid = mo.HankelGrid(700e-6, 5000)
    figure_geometry(out)
    figure_needle(out, grid)
    closed = figure_profiles(out)
    figure_shells(out)
    figure_crosscheck(out)
    (out / "closed_form_widths.json").write_text(json.dumps(closed, indent=1))
    print(json.dumps(closed, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
