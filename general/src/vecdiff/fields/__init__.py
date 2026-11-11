"""Light: beams, sampled fields, the carrier, and the spectra of a region's field.

Every plane-wave representation of a region's field is a ``Spectrum``:
``PlaneSpectrum`` (an FFT lattice), ``LatticeSpectrum`` (carried samples
of a lattice leg), ``HankelSpectrum`` (azimuthal orders of a meridian leg)
and ``BeamSpectrum`` (a beam referred to a chart).
"""

from vecdiff.fields import polarization, profiles
from vecdiff.fields.angular_spectrum import angular_spectrum
from vecdiff.fields.beam_spectrum import BeamSpectrum
from vecdiff.fields.beams import FocusedBeam, PlaneBeam
from vecdiff.fields.carrier import Carrier
from vecdiff.fields.eikonal import EikonalField
from vecdiff.fields.electromagnetic_field import (
    ElectricField,
    ElectromagneticField,
    TransverseElectricField,
)
from vecdiff.fields.hankel_spectrum import HankelSpectrum
from vecdiff.fields.lattice_spectrum import LatticeSpectrum
from vecdiff.fields.spectrum import PlaneSpectrum, PlaneWaveExpansion, Spectrum

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
    "Spectrum",
    "PlaneSpectrum",
    "LatticeSpectrum",
    "HankelSpectrum",
    "BeamSpectrum",
    "PlaneWaveExpansion",
    "angular_spectrum",
]
