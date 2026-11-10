"""Electric and electromagnetic fields: samples, beams, modal and eikonal representations."""

from vecdiff.fields import polarization, profiles
from vecdiff.fields.angular_spectrum import angular_spectrum
from vecdiff.fields.beams import FocusedBeam, PlaneBeam
from vecdiff.fields.carrier import Carrier
from vecdiff.fields.eikonal import EikonalField
from vecdiff.fields.electromagnetic_field import (
    ElectricField,
    ElectromagneticField,
    TransverseElectricField,
)
from vecdiff.fields.spectrum import PlaneSpectrum, PlaneWaveExpansion

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
