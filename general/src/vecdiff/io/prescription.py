"""Readable JSON prescriptions of a path, with explicit SI units and regions."""

import json
from pathlib import Path

from vecdiff.geometry.frames import Frame
from vecdiff.media.medium import Dielectric, HomogeneousRegion, Medium
from vecdiff.surfaces.asphere import EvenAsphere
from vecdiff.surfaces.axisymmetric import SphericalCap
from vecdiff.surfaces.surface import Plane
from vecdiff.system import OpticalSystem, Reflection, Refraction, Stop

SCHEMA = "vecdiff-path/1"


def shape_record(surface):
    if isinstance(surface, Plane):
        return {"type": "plane"}
    if type(surface) is SphericalCap:
        return {"type": "sphere", "radius_m": surface.radius}
    if isinstance(surface, EvenAsphere):
        return {
            "type": "even_asphere",
            "curvature_per_m": surface.curvature,
            "conic": surface.conic,
            "coefficients": list(surface.coefficients),
        }
    raise TypeError(
        "this surface has no supported JSON prescription; callable freeforms "
        "and closed surfaces are not serialized"
    )


def placement(frame):
    return {"origin_m": frame.origin.tolist(), "basis": frame.basis.tolist()}


def element_record(element):
    if isinstance(element, Stop):
        return {"kind": "stop", "radius_m": element.radius, **placement(element.frame)}
    record = {
        "shape": shape_record(element.surface),
        **placement(element.surface.frame),
        "edge_m": element.edge,
    }
    if isinstance(element, Refraction):
        return {
            "kind": "refraction",
            "before": element.before.name,
            "after": element.after.name,
            **record,
        }
    behind = None if element.behind is None else element.behind.name
    return {
        "kind": "reflection",
        "medium": element.medium.name,
        "behind": behind,
        **record,
    }


def material_record(region):
    """A region and its material; an absorbing one's index as n' and kappa."""
    material = region.material
    record = {"name": region.name, "material": material.name}
    if material.absorbing:
        return record | {"n": material.n_real, "kappa": material.extinction}
    return record | {"n": material.n}


def load_material(item):
    if "kappa" in item:
        return Medium(complex(item["n"], item["kappa"]), item["material"])
    return Dielectric(item["n"], item["material"])


def save_system(system: OpticalSystem, path):
    """Serialize a path of plane, spherical-cap and even-asphere encounters and stops."""
    regions = []
    for element in system.elements:
        for region in (element.incident, element.outgoing, element.far):
            if region is not None and region not in regions:
                regions.append(region)
    data = {
        "schema": SCHEMA,
        "length_unit": "m",
        "regions": [material_record(r) for r in regions],
        "elements": [element_record(e) for e in system.elements],
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def load_surface(record):
    frame = Frame(record["origin_m"], record["basis"])
    shape = record["shape"]
    if shape["type"] == "plane":
        return Plane(frame)
    if shape["type"] == "sphere":
        return SphericalCap(shape["radius_m"], frame)
    if shape["type"] == "even_asphere":
        return EvenAsphere(
            shape["curvature_per_m"],
            shape["conic"],
            tuple(shape["coefficients"]),
            frame,
        )
    raise ValueError(f"unsupported prescribed surface: {shape['type']}")


def load_system(path) -> OpticalSystem:
    """Load a prescription without guessing units, surface types or regions."""
    data = json.loads(Path(path).read_text())

    if data.get("schema") != SCHEMA or data.get("length_unit") != "m":
        raise ValueError(f"expected {SCHEMA} with explicit metre units")

    regions = {
        item["name"]: HomogeneousRegion(item["name"], load_material(item))
        for item in data["regions"]
    }
    elements = []
    for record in data["elements"]:
        kind = record["kind"]
        if kind == "stop":
            frame = Frame(record["origin_m"], record["basis"])
            elements.append(Stop(record["radius_m"], frame))
        elif kind == "refraction":
            elements.append(
                Refraction(
                    load_surface(record),
                    regions[record["before"]],
                    regions[record["after"]],
                    edge=record["edge_m"],
                )
            )
        elif kind == "reflection":
            behind = record["behind"]
            elements.append(
                Reflection(
                    load_surface(record),
                    regions[record["medium"]],
                    None if behind is None else regions[behind],
                    edge=record["edge_m"],
                )
            )
        else:
            raise ValueError(f"unsupported path element: {kind}")
    return OpticalSystem(*elements)
