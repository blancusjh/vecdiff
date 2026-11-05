"""Electric and electromagnetic fields: samples, modal and eikonal representations."""

from vecdiff_general.fields.angular_spectrum import angular_spectrum
from vecdiff_general.fields.eikonal import EikonalField
from vecdiff_general.fields.electromagnetic_field import (
    ElectricField,
    ElectromagneticField,
    TransverseElectricField,
)
from vecdiff_general.fields.spectrum import PlaneSpectrum, PlaneWaveExpansion

__all__ = [
    "ElectricField",
    "ElectromagneticField",
    "TransverseElectricField",
    "EikonalField",
    "PlaneSpectrum",
    "PlaneWaveExpansion",
    "angular_spectrum",
]
