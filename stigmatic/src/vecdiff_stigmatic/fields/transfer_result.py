"""Inspectable geometric fields, local Fresnel traces and ray-power accounting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._arrays import frozen_array
from ..geometry.rays import RayBundle
from .electromagnetic_field import ElectromagneticField


@dataclass(frozen=True, eq=False)
class TransferResult:
    """Geometric fields and local Fresnel traces, not a solved curved boundary."""

    rays: RayBundle
    entrance: ElectromagneticField
    exit: ElectromagneticField
    incident_surface: ElectromagneticField
    reflected_surface: ElectromagneticField
    transmitted_surface: ElectromagneticField
    rs: np.ndarray
    rp: np.ndarray
    ts: np.ndarray
    tp: np.ndarray

    def __post_init__(self):
        for name in ("rs", "rp", "ts", "tp"):
            object.__setattr__(
                self, name, frozen_array(getattr(self, name), dtype=complex)
            )

    def power_budget(self):
        incident = self.entrance.power()
        reflected = -self.reflected_surface.power()
        transmitted = self.exit.power()

        return {
            "incident_W": incident,
            "reflected_W": reflected,
            "transmitted_W": transmitted,
            "balance_W": incident - reflected - transmitted,
        }
