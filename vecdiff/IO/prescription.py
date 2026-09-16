"""Strict native CSV interoperation with geometrical-raytracer.

    Radius, thickness, aperture, vertex and polynomial units are millimetres.
    Refractive indices are tabulated at an explicitly selected vacuum wavelength;
    neither dispersion interpolation nor an assumed glass catalogue is hidden here.
"""
import csv
import re
from pathlib import Path
import numpy as np
from ..geometry.frames import Frame
from ..media.medium import Medium
from ..surfaces.asphere import EvenAsphere
from ..surfaces.surface import Plane
from ..interfaces.optical_system import OpticalSystem, SurfaceEncounter

COEFFICIENTS = [f"C{j}_mm^-{2*j+1}" for j in range(1, 7)]


def read_prescription(path, *, wavelength_nm=None, incident_index=1., vertex_tolerance_mm=1e-6):
    """Read the native CSV dialect, preserving folds, stops and all encounters.

    Multiple wavelength columns require explicit selection. Negative thickness
    does not itself change propagation direction: reflection does. In particular,
    a virtual stop may lie behind the preceding vertex in a forward branch.
    """
    if not np.isfinite(vertex_tolerance_mm) or vertex_tolerance_mm < 0:
        raise ValueError("vertex tolerance must be finite and nonnegative")
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        required = {"surface", "surface_type", "radius_mm", "thickness_to_next_mm", "material_after"}
        if not required.issubset(headers) or len(headers) != len(set(headers)):
            raise ValueError("missing or duplicate prescription columns")
        columns = {float(m[1]): h for h in headers if (m := re.fullmatch(r"index_([0-9.]+)_nm", h))}
        if wavelength_nm is None:
            if len(columns) != 1: raise ValueError("select wavelength_nm when the table does not contain exactly one wavelength")
            wavelength_nm = next(iter(columns))
        if wavelength_nm not in columns or wavelength_nm <= 0:
            raise ValueError("requested wavelength has no tabulated index column")
        column = columns[wavelength_nm]
        medium, z, direction, encounters = Medium(incident_index), 0., 1, []
        for row in reader:
            number = int(row["surface"])
            if number != len(encounters)+1: raise ValueError("surface numbers must be consecutive starting at one")
            def number_value(key, default=None):
                token = (row.get(key) or "").strip()
                if not token:
                    if default is None: raise ValueError(f"missing {key} at surface {number}")
                    return default
                value = float(token)
                if not np.isfinite(value): raise ValueError(f"nonfinite {key} at surface {number}")
                return value
            radius = number_value("radius_mm")
            thickness = number_value("thickness_to_next_mm")
            kind = row["surface_type"].strip().lower()
            kinds = {"refractive": "refract", "plane": "refract", "mirror": "reflect", "aperture_stop": "stop"}
            if kind not in kinds: raise ValueError(f"unsupported surface_type {kind!r}")
            material = row["material_after"].strip()
            if (material.upper() == "REFL") != (kind == "mirror"):
                raise ValueError(f"inconsistent mirror/material at surface {number}")
            after = medium if kind == "mirror" else Medium(number_value(column))
            if kind == "aperture_stop" and after != medium:
                raise ValueError("an aperture stop cannot silently change medium")
            semidiameter = number_value("clear_semidiameter_mm") if (row.get("clear_semidiameter_mm") or "").strip() else None
            if semidiameter is not None and semidiameter <= 0: raise ValueError("clear semidiameter must be positive")
            if (row.get("vertex_z_mm") or "").strip() and abs(number_value("vertex_z_mm")-z) > vertex_tolerance_mm:
                raise ValueError(f"vertex disagrees with accumulated thickness at surface {number}")
            conic = number_value("K", 0.)
            coefficients = tuple(number_value(key, 0.) for key in COEFFICIENTS)
            frame = Frame(origin=[0., 0., z])
            surface = (Plane(frame) if radius == 0 and not any(coefficients)
                       else EvenAsphere(0. if radius == 0 else 1/radius, conic, coefficients, frame))
            if kind == "plane" and (radius != 0 or any(coefficients)):
                raise ValueError("a plane row cannot contain curvature or aspheric coefficients")
            if semidiameter is not None and hasattr(surface, "sag"): surface.sag(semidiameter)
            encounters.append(SurfaceEncounter(number, surface, kinds[kind], medium, after,
                                               semidiameter, thickness, direction, material))
            medium = after
            if kind == "mirror": direction *= -1
            z += thickness
    if not encounters: raise ValueError("empty optical prescription")
    return OpticalSystem(tuple(encounters), wavelength_nm*1e-6, name=Path(path).stem)


def write_prescription(system, path):
    """Write a lossless native CSV round trip (17 significant decimal digits)."""
    if system.length_unit != "mm": raise ValueError("native CSV requires millimetres")
    column = f"index_{system.wavelength*1e6:.17g}_nm"
    headers = ["surface", "surface_type", "radius_mm", "thickness_to_next_mm", "material_after", column,
               "clear_semidiameter_mm", "aspheric", "K", *COEFFICIENTS, "vertex_z_mm"]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers); writer.writeheader()
        for e in system.encounters:
            s = e.surface
            if not isinstance(s, (Plane, EvenAsphere)) or not np.allclose(s.frame.rotation, np.eye(3)) or np.any(s.frame.origin[:2]):
                raise ValueError("native CSV supports only coaxial planes and even aspheres")
            c, conic, coeff = getattr(s, "curvature", 0.), getattr(s, "conic", 0.), getattr(s, "coefficients", ())
            if len(coeff) > 6: raise ValueError("native CSV supports six polynomial coefficients")
            kind = {"refract": "plane" if isinstance(s, Plane) else "refractive", "reflect": "mirror", "stop": "aperture_stop"}[e.interaction]
            f = lambda v: format(v, ".17g")
            writer.writerow(dict(surface=e.number, surface_type=kind, radius_mm=f(1/c if c else 0),
                thickness_to_next_mm=f(e.thickness), material_after=e.material_after,
                clear_semidiameter_mm="" if e.semidiameter is None else f(e.semidiameter),
                aspheric=bool(conic or any(coeff)), K=f(conic), vertex_z_mm=f(s.frame.origin[2]),
                **{column:f(e.transmitted_medium.n)}, **{key:f(coeff[j] if j < len(coeff) else 0.) for j,key in enumerate(COEFFICIENTS)}))
