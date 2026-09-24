"""Readable JSON prescriptions with explicit SI units and material regions."""

import json
from pathlib import Path

from ..geometry.frames import Frame
from ..interfaces.dielectric_interface import DielectricInterface
from ..interfaces.optical_system import OpticalSystem
from ..media.medium import Dielectric, HomogeneousRegion
from ..surfaces.asphere import EvenAsphere
from ..surfaces.axisymmetric import SphericalCap
from ..surfaces.surface import Plane


def save_system(system: OpticalSystem, path):
    """Serialize supported placed plane, spherical and even-asphere encounters."""
    regions = (system.interfaces[0].minus,) + tuple(
        interface.plus for interface in system.interfaces
    )
    records = []

    for interface, aperture, stop in zip(
        system.interfaces, system.apertures, system.stops
    ):
        surface = interface.surface

        if isinstance(surface, Plane):
            shape = {"type": "plane"}
        elif isinstance(surface, SphericalCap):
            shape = {"type": "sphere", "radius_m": surface.radius}
        elif isinstance(surface, EvenAsphere):
            shape = {
                "type": "even_asphere",
                "curvature_per_m": surface.curvature,
                "conic": surface.conic,
                "coefficients": list(surface.coefficients),
            }
        else:
            raise TypeError(
                "this surface has no supported JSON prescription; callable freeforms are not serialized"
            )

        records.append(
            {
                "shape": shape,
                "origin_m": surface.frame.origin.tolist(),
                "basis": surface.frame.basis.tolist(),
                "aperture_m": aperture,
                "stop_m": stop,
            }
        )

    data = {
        "schema": "logos-optics/1",
        "length_unit": "m",
        "regions": [
            {
                "name": region.name,
                "material": region.material.name,
                "n": region.material.n,
            }
            for region in regions
        ],
        "interfaces": records,
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def load_system(path) -> OpticalSystem:
    """Load a prescription without guessing units, surface types or ray direction."""
    data = json.loads(Path(path).read_text())

    if data.get("schema") != "logos-optics/1" or data.get("length_unit") != "m":
        raise ValueError("expected logos-optics/1 with explicit metre units")

    regions = tuple(
        HomogeneousRegion(item["name"], Dielectric(item["n"], item["material"]))
        for item in data["regions"]
    )

    if len(regions) != len(data["interfaces"]) + 1:
        raise ValueError("prescription needs one more region than interface")

    interfaces, apertures, stops = [], [], []

    for index, record in enumerate(data["interfaces"]):
        frame = Frame(record["origin_m"], record["basis"])
        shape = record["shape"]

        if shape["type"] == "plane":
            surface = Plane(frame)
        elif shape["type"] == "sphere":
            surface = SphericalCap(shape["radius_m"], frame)
        elif shape["type"] == "even_asphere":
            surface = EvenAsphere(
                shape["curvature_per_m"],
                shape["conic"],
                tuple(shape["coefficients"]),
                frame,
            )
        else:
            raise ValueError(f"unsupported prescribed surface: {shape['type']}")

        interfaces.append(
            DielectricInterface(surface, regions[index], regions[index + 1])
        )
        apertures.append(record["aperture_m"])
        stops.append(record.get("stop_m"))

    return OpticalSystem(tuple(interfaces), tuple(apertures), tuple(stops))
