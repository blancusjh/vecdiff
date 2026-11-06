"""Electric and electromagnetic fields: samples, beams, modal and eikonal representations."""

from vecdiff_general.fields import polarization, profiles
from vecdiff_general.fields.angular_spectrum import angular_spectrum
from vecdiff_general.fields.beams import FocusedBeam, PlaneBeam
from vecdiff_general.fields.carrier import Carrier
from vecdiff_general.fields.eikonal import EikonalField
from vecdiff_general.fields.electromagnetic_field import (
    ElectricField,
    ElectromagneticField,
    TransverseElectricField,
)
from vecdiff_general.fields.spectrum import PlaneSpectrum, PlaneWaveExpansion

__all__ = [
    "PlaneBeam",
    "FocusedBeam",
    "Carrier",
    "profiles",
    "polarization",
    "ElectricField",
    "ElectromagneticField",
    "TransverseElectricField",
    "EikonalField",
    "PlaneSpectrum",
    "PlaneWaveExpansion",
    "angular_spectrum",
]
