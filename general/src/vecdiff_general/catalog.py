"""Published optical prescriptions, expressed in metres.

AL1225: Thorlabs V21 catalogue, printed p. 744 (asphere coefficients),
https://www.thorlabs.com/images/Catalog/V21/V21_3_Optics.pdf
The catalogue draws the asphere as an exit face (negative curvature).
Here it faces an incoming collimated beam: reverse ALL sag coefficients.
The positive-curvature prescription is also tabulated in Table 2 of
https://vccimaging.org/Publications/Jimenez2024E2EFLuidic/Jimenez2024E2EFLuidic_supp.pdf

N-BK7 dispersion: SCHOTT N-BK7 data sheet, Sellmeier coefficients,
https://www.schott.com/en-gb/products/optical-glass
"""

import numpy as np

from vecdiff_general.geometry import Frame
from vecdiff_general.media import Dielectric, HomogeneousRegion
from vecdiff_general.surfaces import EvenAsphere, Plane
from vecdiff_general.system import DielectricInterface, OpticalSystem


def thorlabs_al1225(*, wavelength0=780e-9, scale=1.0):
    """Uncoated AL1225, aspheric entrance at z=0 and plane exit at z=4 mm.

    The physical part has diameter 12.5 mm, nominal EFL 25 mm, NA 0.23,
    and design wavelength 780 nm. V21 specifies clear diameters of 11.25 mm
    (aspheric face) and 10.42 mm (plane); these are specified optical zones,
    not opaque stops. Both material edges have radius 6.25 mm.

    ``scale`` scales every geometric length, including the aspheric terms;
    scale != 1 is a scaled model, not the manufactured part. Wavelength is
    in metres and selects lossless N-BK7's index via SCHOTT's Sellmeier law.
    Air is assigned n=1. Coatings, absorption and manufacturing errors are
    not included. The prescription contains finite edges; the current
    forward PropagationPipeline rejects them rather than silently removing
    them. A continued-face approximation must be constructed explicitly.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if not np.isfinite(wavelength0) or not 300e-9 <= wavelength0 <= 2500e-9:
        raise ValueError("N-BK7 wavelength must be between 300 and 2500 nm")
    wavelength_um2 = (wavelength0 * 1e6) ** 2
    b = np.array([1.03961212, 0.231792344, 1.010469450])
    c = np.array([0.00600069867, 0.0200179144, 103.5606530])
    index = float(np.sqrt(1 + np.sum(b * wavelength_um2 / (wavelength_um2 - c))))
    air_before = HomogeneousRegion("air before AL1225", Dielectric(1.0, "air"))
    glass = HomogeneousRegion("AL1225 glass", Dielectric(index, "SCHOTT N-BK7"))
    air_after = HomogeneousRegion("air after AL1225", Dielectric(1.0, "air"))
    mm = 1e-3 * scale
    # A_(2j) in mm**(1-2j), for the positive-curvature entrance orientation.
    coefficients_mm = (
        1.8429898e-6,
        -3.8172252e-9,
        -2.4345457e-11,
        3.1730496e-14,
        -3.700237e-15,
        6.5107821e-17,
        -4.9604147e-19,
    )
    entrance = EvenAsphere(
        curvature=1 / (12.78 * mm),
        conic=-0.6,
        coefficients=tuple(
            value / mm ** (2 * order - 1)
            for order, value in enumerate(coefficients_mm, start=2)
        ),
    )
    exit_face = Plane(Frame([0, 0, 4 * mm]))
    return OpticalSystem(
        (
            DielectricInterface(entrance, air_before, glass),
            DielectricInterface(exit_face, glass, air_after),
        ),
        apertures=(6.25 * mm, 6.25 * mm),
    )
