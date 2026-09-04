"""Independent historical direct-radiation implementation for comparison.

Prescribed-current radiation does not determine the correct dielectric
boundary currents. This distinction also applies to the new native method.
"""
from .legacy.vecdiff.reference.stratton_chu import franz_integral

__all__ = ["franz_integral"]
