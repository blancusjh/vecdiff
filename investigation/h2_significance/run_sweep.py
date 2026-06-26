import csv
import itertools
from pathlib import Path
import warnings

import numpy as np

from vecdiff.fresnel import FresnelOvoid, FresnelOvoidParax
from vecdiff.hankel import HankelTransform


LAM = 532e-6
N0 = 1.0
VARPHI = 0.0
PHASE = exp(2.0j * VARPHI)

N = 900
FOCUS_POWER = 2.2

OUT_DIR = Path(__file__).resolve().parent
OUT_CSV = OUT_DIR / "results.csv"


def make_grid(max_val, n=N, power=FOCUS_POWER):
    u = np.linspace(0.0, 1.0, n)
    return max_val * u**power


def compute_case(R, a_frac, ni, z0, zi, exact=True):
    a = a_frac * R
    qmax = ni / (LAM * zi) * R

    r = make_grid(R)
    q = make_grid(qmax)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if exact:
                F = FresnelOvoid(n0=N0, z0=z0, ni=ni, zi=zi)
                ts = F.ts(r)
                tp = F.tp(r)
            else:
                F = FresnelOvoidParax(N0, ni, z0, zi)
                ts, tp = F.coeffs(r)
    except Exception:
        return None

    # Exact model can produce an indeterminate value at r=0; repair by interpolation.
    ts = np.asarray(ts, dtype=float)
    tp = np.asarray(tp, dtype=float)
    if not np.all(np.isfinite(ts)):
        bad = ~np.isfinite(ts)
        good = ~bad
        if np.count_nonzero(good) < 2:
            return None
        ts[bad] = np.interp(r[bad], r[good], ts[good])
    if not np.all(np.isfinite(tp)):
        bad = ~np.isfinite(tp)
        good = ~bad
        if np.count_nonzero(good) < 2:
            return None
        tp[bad] = np.interp(r[bad], r[good], tp[good])

    if not (np.all(np.isfinite(ts)) and np.all(np.isfinite(tp))):
        return None

    EL = np.ones_like(r, dtype=complex)
    ER = np.ones_like(r, dtype=complex)
    ap = (r <= a).astype(float)
    EL *= ap
    ER *= ap

    HT = HankelTransform.transform_array

    C_L_H0 = 2.0 * π * HT(0, r, np.real((tp + ts) * EL), q)
    C_L_H1 = -2.0 * π * PHASE * HT(1, r, np.real((tp - ts) * ER), q)
    C_R_H0 = 2.0 * π * HT(0, r, np.real((tp + ts) * ER), q)
    C_R_H2 = -2.0 * π * PHASE * HT(2, r, np.real((tp - ts) * EL), q)

    E_L = C_L_H0 + C_L_H1
    E_R = C_R_H0 + C_R_H2

    I_full = np.abs(E_L) ** 2 + np.abs(E_R) ** 2
    I_noH2 = np.abs(E_L) ** 2 + np.abs(C_R_H0) ** 2

    # Metrics
    peak_ratio_R = float(np.max(np.abs(C_R_H2)) / (np.max(np.abs(C_R_H0)) + 1e-15))
    energy_ratio_R = float(np.trapezoid(np.abs(C_R_H2) ** 2 * q, q) / (np.trapezoid(np.abs(C_R_H0) ** 2 * q, q) + 1e-15))

    peak_ratio_L = float(np.max(np.abs(C_L_H1)) / (np.max(np.abs(C_L_H0)) + 1e-15))

    intensity_delta_rel = float(
        np.trapezoid(np.abs(I_full - I_noH2) * q, q)
        / (np.trapezoid(I_full * q, q) + 1e-15)
    )

    result = {
        "model": "exact" if exact else "paraxial",
        "R": R,
        "a": a,
        "a_over_R": a_frac,
        "ni": ni,
        "z0": z0,
        "zi": zi,
        "qmax": qmax,
        "peak_ratio_R_H2_over_H0": peak_ratio_R,
        "energy_ratio_R_H2_over_H0": energy_ratio_R,
        "peak_ratio_L_H1_over_H0": peak_ratio_L,
        "intensity_delta_from_H2": intensity_delta_rel,
    }
    if not np.all(np.isfinite(list(result.values())[8:])):
        return None
    return result


def main():
    Rs = [0.8, 1.0, 1.2, 1.6]
    a_fracs = [0.7, 0.85, 0.95]
    nis = [1.5, 1.8, 2.0]
    z0s = [2.5, 5.0]
    zis = [5.0, 10.0]

    rows = []
    for exact in [True, False]:
        for R, a_frac, ni, z0, zi in itertools.product(Rs, a_fracs, nis, z0s, zis):
            if zi <= z0:
                continue
            result = compute_case(R, a_frac, ni, z0, zi, exact=exact)
            if result is not None:
                rows.append(result)

    rows.sort(key=lambda d: d["intensity_delta_from_H2"], reverse=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    print("Top 12 by intensity_delta_from_H2:")
    for i, r in enumerate(rows[:12], start=1):
        print(
            i,
            r["model"],
            f"R={r['R']}",
            f"a/R={r['a_over_R']}",
            f"ni={r['ni']}",
            f"z0={r['z0']}",
            f"zi={r['zi']}",
            f"deltaI={r['intensity_delta_from_H2']:.4f}",
            f"ER_energy(H2/H0)={r['energy_ratio_R_H2_over_H0']:.4f}",
            f"ER_peak(H2/H0)={r['peak_ratio_R_H2_over_H0']:.4f}",
        )


if __name__ == "__main__":
    main()
