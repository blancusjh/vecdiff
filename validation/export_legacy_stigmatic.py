"""Export a preservation fixture from an immutable legacy snapshot.

Run with the legacy environment (which includes matplotlib), for example:
    python -B validation/export_legacy_stigmatic.py --source /path/to/7a1a235

The normal test suite only reads the resulting NPZ; it never imports vecdiff.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "reference_data/legacy_stigmatic.npz",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from vecdiff import CartesianSurface
    from vecdiff.reference import focal_field_reference
    from vecdiff.transfer import interface_operator

    surface = CartesianSurface(n0=1.0, ni=1.5, z0=-10.0, zi=6.0)
    aperture = 0.95 * surface.aperture_limit
    nodes, _ = np.polynomial.legendre.leggauss(48)
    radius = aperture * (nodes + 1) / 2
    operator = interface_operator(surface, radius)
    geometry = operator.geom
    wavelength = 0.000532  # mm in the legacy code; exported coordinates are SI.
    x = np.linspace(-2, 2, 41) * wavelength
    points = np.column_stack((x, np.zeros_like(x), np.full_like(x, 6.0)))
    _, green = focal_field_reference(
        surface,
        wavelength,
        np.ones_like,
        aperture=aperture,
        observation=points,
        n_r=320,
        n_phi=96,
        chunk=1000,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        radius_m=radius * 1e-3,
        sag_m=geometry.z * 1e-3,
        geometric_amplitude=geometry.A,
        tp=operator.t_p,
        ts=operator.t_s,
        aperture_m=aperture * 1e-3,
        wavelength0_m=wavelength * 1e-3,
        points_m=points * 1e-3,
        green_electric=green,
    )
    provenance = {
        "revision": "7a1a235",
        "branch": "claude/transfer-matrix-geometric-factor-664h9g",
        "source": str(args.source.resolve()),
        "coordinates": "metres after explicit mm conversion",
        "field": "complex Cartesian E, legacy numerical source amplitude 1 interpreted as 1 V/m",
        "geometry": "n0=1, ni=1.5, object=-10 mm, image=6 mm; 95% grazing aperture",
        "reference": "legacy full Green radiation of physical-optics currents",
        "quadrature": {"radial": 320, "azimuthal": 96},
        "claim": "geometry/transfer preservation and Debye asymptotic comparison; not exact dielectric scattering",
        "excluded": "legacy Debye helper with the audited longitudinal sign discrepancy",
    }
    args.output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
