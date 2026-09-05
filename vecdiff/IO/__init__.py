"""Optical prescription import/export. No ray tracer is required at runtime."""
from .prescription import read_prescription, write_prescription

__all__ = ["read_prescription", "write_prescription"]
