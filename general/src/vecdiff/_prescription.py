"""The optical system as a lens prescription: one row per element of its path.

``rows(system)`` is the data, each element as an optical designer reads it:
what it does to the light, its surface's kind and shape, where its vertex
lies and how it is tilted, the media on its two sides, its aperture and the
distance to the next element. ``text``, ``html`` and ``markdown`` lay those
rows out with the regions of the path's segments and the notes a row cannot
hold. Nothing is printed that the system does not hold: a surface without
a radius (a freeform, a sag function) says what it is instead.
"""

import math
import textwrap

from vecdiff._backend import to_numpy
from vecdiff._printing import (
    html_table,
    markdown_table,
    material_name,
    number,
    shorten,
    text_table,
    vector,
)

# Transverse offsets and tilts below this, relative to the system's size, are roundoff.
COAXIAL = 1e-12
# Notes under the text table wrap at this width, the widest coaxial table.
WIDTH = 110


def unwrapped(surface):
    """The surface a reversed view shows, and whether the view is reversed."""
    from vecdiff.surfaces.surface import ReversedSurface

    reversed_ = False
    while isinstance(surface, ReversedSurface):
        surface, reversed_ = surface.base, not reversed_
    return surface, reversed_


def kind(element):
    """What the element does to the light: Refraction, Reflection, Mirror or Stop."""
    from vecdiff.system import Reflection, Stop

    if isinstance(element, Stop):
        return "Stop"
    if isinstance(element, Reflection):
        return "Mirror" if element.behind is None else "Reflection"
    return "Refraction"


def frame_of(element):
    """The frame placing the element: its screen's, or its surface's as built."""
    if kind(element) == "Stop":
        return element.frame
    return unwrapped(element.surface)[0].frame


def shape(surface):
    """The kind, local signed radius, conic, coefficients and notes of a surface."""
    from vecdiff.surfaces import (
        AxisymmetricSurface,
        CartesianOval,
        CylindricalCap,
        EvenAsphere,
        FreeformSurface,
        Plane,
        SphericalCap,
    )
    from vecdiff.surfaces.surface import ImplicitSurface

    base, reversed_ = unwrapped(surface)
    name = type(base).__name__
    entry = dict(
        type=name + (" (closed)" if isinstance(base, ImplicitSurface) else ""),
        reversed=reversed_,
        radius=None,
        form=None,
        conic=None,
        coefficients=(),
        notes=[],
    )
    if reversed_:
        entry["type"] = "reversed " + entry["type"]
    if isinstance(base, Plane):
        entry["form"] = "plane"
    elif isinstance(base, (SphericalCap, CylindricalCap)):
        entry["radius"] = float(base.radius)
        if isinstance(base, CylindricalCap):
            entry["notes"].append(f"cylinder axis along {vector(base.frame.v)}")
    elif isinstance(base, EvenAsphere):
        c = float(base.curvature)
        entry["radius"] = 1 / c if c else None
        entry["form"] = None if c else "plane"
        entry["conic"] = float(base.conic) if base.conic else None
        entry["coefficients"] = tuple(base.coefficients)
    elif isinstance(base, CartesianOval):
        c = float(base.design.vertex_curvature)
        entry["radius"] = 1 / c if c else None
        entry["form"] = None if c else "plane"
        entry["notes"].append(
            f"Cartesian oval: object at z = {number(base.object_z)}, image at "
            f"z = {number(base.image_z)} from its vertex, design indices "
            f"{number(base.n_object)} -> {number(base.n_image)}; R is the "
            "vertex radius"
        )
    elif isinstance(base, AxisymmetricSurface):
        entry["form"] = "sag(r)"
        entry["notes"].append(f"sag function {_name(base.sag_function)}")
    elif isinstance(base, FreeformSurface):
        entry["form"] = "freeform"
        entry["notes"].append(f"height function {_name(base.height_function)}")
    else:
        entry["form"] = "graph"
    return entry


def _name(function):
    from vecdiff._printing import name_of

    return name_of(function)


def _vector(array):
    return tuple(float(x) for x in to_numpy(array).ravel())


def is_coaxial(system):
    """Whether every element's vertex lies on the z axis with its normal along +-z."""
    frames = [frame_of(element) for element in system.elements]
    size = max([1.0] + [max(abs(x) for x in _vector(f.origin)) for f in frames])
    for frame in frames:
        x, y, _ = _vector(frame.origin)
        nx, ny, _ = _vector(frame.normal)
        if max(abs(x), abs(y)) > COAXIAL * size or max(abs(nx), abs(ny)) > COAXIAL:
            return False
    return True


def _media(element):
    """The regions on the element's two sides, as (before, after, behind)."""
    label = kind(element)
    if label == "Refraction":
        return element.before, element.after, None
    if label == "Reflection":
        return element.medium, element.medium, element.behind
    return element.medium, element.medium, None


def rows(system):
    """One dict per element of the path: the prescription as plain data.

    In a coaxial system ``radius`` and ``coefficients`` are signed along +z
    (R > 0: centre of curvature beyond the vertex) and ``thickness`` is the
    signed axial distance to the next vertex; otherwise radii are signed
    along each surface's own normal and ``thickness`` is the distance
    between consecutive vertices. Lengths are in the system's units.
    """
    coaxial = is_coaxial(system)
    labels, table = [], []
    for index, element in enumerate(system.elements, 1):
        label, first = _label(labels, element, index)
        frame = frame_of(element)
        origin, normal = _vector(frame.origin), _vector(frame.normal)
        entry = (
            dict(type="screen", radius=None, form=None, conic=None)
            | dict(coefficients=(), notes=[], reversed=False)
            if kind(element) == "Stop"
            else shape(element.surface)
        )
        sign = (1.0 if normal[2] > 0 else -1.0) if coaxial else 1.0
        before, after, behind = _media(element)
        table.append(
            dict(
                index=index,
                element=kind(element),
                surface=label,
                repeated=first is not None,
                first_met=first,
                type=entry["type"],
                vertex=origin,
                normal=normal,
                tilt=math.degrees(math.acos(max(-1.0, min(1.0, normal[2])))),
                radius=None if entry["radius"] is None else sign * entry["radius"],
                form=entry["form"],
                conic=entry["conic"],
                coefficients=tuple(sign * a for a in entry["coefficients"]),
                before=material_name(before),
                after=material_name(after),
                behind=material_name(behind),
                regions=(_region(before), _region(after), _region(behind)),
                n_before=_index(before),
                n_after=_index(after),
                n_behind=_index(behind),
                aperture=element.aperture,
                thickness=None,
                notes=entry["notes"],
            )
        )
    for here, there in zip(table, table[1:]):
        step = [b - a for a, b in zip(here["vertex"], there["vertex"])]
        here["thickness"] = step[2] if coaxial else math.hypot(*step)
        if _closed_again(here, there):
            # Both meet one closed surface, perhaps on opposite sides of it:
            # the distance the light travels is not the vertices' distance.
            here["thickness"] = None
    return table


def _closed_again(here, there):
    """Whether two consecutive rows meet the same closed surface."""
    return (
        here["surface"] is not None
        and here["surface"] == there["surface"]
        and "(closed)" in here["type"]
    )


def _label(labels, element, index):
    """The surface's label (S1, S2, ...) and the element that met it first, if any."""
    if kind(element) == "Stop":
        return None, None
    for number_, (surface, first) in enumerate(labels, 1):
        if surface is element.surface or surface == element.surface:
            return f"S{number_}", first
    labels.append((element.surface, index))
    return f"S{len(labels)}", None


def _region(medium):
    return None if medium is None else medium.name


def _index(medium):
    if medium is None:
        return None
    material = getattr(medium, "material", medium)
    return material.n


# --- Layout -----------------------------------------------------------------


def columns(system, table):
    """Headers, alignments and cell rows for the system's table."""
    coaxial = is_coaxial(system)
    decentred = any(max(map(abs, r["vertex"][:2])) > 0 for r in table)
    tilted = any(r["tilt"] > 0 for r in table)
    conic = any(r["conic"] is not None or r["coefficients"] for r in table)
    spec = [
        ("#", "r", lambda r: str(r["index"])),
        ("element", "l", lambda r: r["element"]),
        ("surface", "l", _surface_cell),
        ("type", "l", lambda r: r["type"]),
        ("z", "n", lambda r: number(r["vertex"][2])),
    ]
    if not coaxial and decentred:
        spec.append(("decentre (x, y)", "l", lambda r: vector(r["vertex"][:2])))
    if not coaxial and tilted:
        spec.append(("tilt", "n", lambda r: number(r["tilt"]) + "°"))
        spec.append(("normal", "l", lambda r: vector(r["normal"])))
    spec.append(("radius", "n", _radius_cell))
    if conic:
        spec.append(("conic, asph.", "l", _conic_cell))
    spec += [
        ("medium", "l", _medium_cell),
        ("n", "l", _index_cell),
        ("aperture r", "n", lambda r: _optional(r["aperture"])),
        (
            "thickness" if coaxial else "to next",
            "n",
            lambda r: _optional(r["thickness"]),
        ),
    ]
    headers = [name for name, _, _ in spec]
    aligns = "".join(align for _, align, _ in spec)
    cells = [[cell(r) for _, _, cell in spec] for r in table]
    return headers, aligns, cells


def _optional(value):
    return "" if value is None else number(value)


def _surface_cell(row):
    if row["surface"] is None:
        return ""
    return row["surface"] + ("*" if row["repeated"] else "")


def _radius_cell(row):
    if row["radius"] is None:
        return row["form"] or ""
    return number(row["radius"])


def _conic_cell(row):
    parts = [] if row["conic"] is None else [f"k {number(row['conic'])}"]
    count = len(row["coefficients"])
    if count:
        parts.append("A4" if count == 1 else f"A4..A{2 * count + 2}")
    return ", ".join(parts)


def _medium_cell(row):
    if row["element"] == "Refraction":
        return f"{row['before']} -> {row['after']}"
    if row["element"] == "Reflection":
        return f"{row['before']}, behind {row['behind']}"
    return "" if row["before"] is None else str(row["before"])


def _index_cell(row):
    if row["n_before"] is None:
        return ""
    if row["element"] == "Refraction":
        return f"{number(row['n_before'])} -> {number(row['n_after'])}"
    if row["element"] == "Reflection":
        return f"{number(row['n_before'])}, behind {number(row['n_behind'])}"
    return number(row["n_before"])


def heading(system):
    """One line: how many elements, whether coaxial, where the light arrives and leaves."""
    count = len(system.elements)
    first, last = system.regions[0], system.regions[-1]
    if count == 0:
        where = "" if first is None else f", light in {_describe(first)}"
        return f"OpticalSystem: no elements{where}"
    axis = ", coaxial along z" if is_coaxial(system) else ""
    plural = "element" if count == 1 else "elements"
    return (
        f"OpticalSystem: {count} {plural}{axis}; light arrives in "
        f"{_describe(first)} and leaves in {_describe(last)}"
    )


def _describe(region):
    if region is None:
        return "no medium"
    return f"{material_name(region)} (n {number(region.material.n)})"


def notes(system, table):
    """What the table's rows cannot hold, one line each."""
    lines = []
    if table:
        lines.append(
            "R > 0: centre of curvature toward +z; thickness: axial distance to "
            "the next vertex; unit of wavelength0."
            if is_coaxial(system)
            else "R signed along each surface normal; to next: distance between "
            "vertices; unit of wavelength0."
        )
    if any(
        region is not None and region.name != material_name(region)
        for region in system.regions
    ):
        lines.append("Regions of the segments: " + _segments(system))
    kinds = {row["element"] for row in table}
    if "Mirror" in kinds:
        lines.append("Mirror: an ideal mirror, on which tangential E vanishes.")
    if "Reflection" in kinds:
        lines.append(
            "Reflection: the Fresnel reflection of the boundary with the medium behind."
        )
    if any(_closed_again(a, b) for a, b in zip(table, table[1:])):
        lines.append(
            "A closed surface is met where the light crosses it, on either side: "
            "z is the vertex of its chart, and the thickness between two meetings "
            "of it is left blank."
        )
    for row in table:
        lines += _row_notes(row)
    return lines


def _segments(system):
    return ", ".join(
        f"[{i}] {'none' if region is None else region.name}"
        for i, region in enumerate(system.regions)
    )


def _row_notes(row):
    lines = []
    tag = f"{row['index']} ({row['surface']})" if row["surface"] else str(row["index"])
    if row["repeated"]:
        lines.append(
            f"* element {row['index']} meets {row['surface']} again, "
            f"first met by element {row['first_met']}"
        )
    if row["coefficients"]:
        terms = ", ".join(
            f"A{2 * j} = {number(a)}" for j, a in enumerate(row["coefficients"], 2)
        )
        lines.append(f"element {tag}: {terms}")
    if not row["repeated"]:
        lines += [f"element {tag}: {note}" for note in row["notes"]]
    return lines


def text(system):
    """The system as plain text: heading, table, notes."""
    table = rows(system)
    parts = [heading(system)]
    if table:
        headers, aligns, cells = columns(system, table)
        parts.append(text_table(headers, cells, aligns))
    parts += [
        textwrap.fill(line, WIDTH, subsequent_indent="  ")
        for line in notes(system, table)
    ]
    return "\n".join(parts)


def html(system):
    """The system as an HTML table with its heading as caption and notes below."""
    table = rows(system)
    headers, aligns, cells = columns(system, table) if table else ([], "", [])
    return html_table(
        headers, cells, aligns, caption=heading(system), notes=notes(system, table)
    )


def markdown(system):
    """The system as a Markdown table with its heading and notes."""
    table = rows(system)
    parts = [f"**{heading(system)}**", ""]
    if table:
        headers, aligns, cells = columns(system, table)
        parts += [markdown_table(headers, cells, aligns), ""]
    parts += [f"- {line}" for line in notes(system, table)]
    return "\n".join(parts)


def summary(system):
    """One line: the elements' kinds and surfaces, the media at both ends."""
    if not system.elements:
        region = system.regions[0]
        where = "" if region is None else f"; in {material_name(region)!r}"
        return f"OpticalSystem(no elements{where})"
    names = []
    for row in rows(system):
        if row["element"] == "Stop":
            names.append("Stop")
        else:
            names.append(f"{row['element']}({row['type']})")
    ends = " -> ".join(
        "none" if r is None else repr(material_name(r))
        for r in (system.regions[0], system.regions[-1])
    )
    return shorten(f"OpticalSystem({', '.join(names)}; {ends})")


def encounter(element):
    """An element in a few words: 'Refraction at SphericalCap, air -> glass'."""
    label = kind(element)
    if label == "Stop":
        where = (
            "" if element.medium is None else f", in {material_name(element.medium)}"
        )
        return f"Stop of radius {number(element.radius)}{where}"
    surface = shape(element.surface)["type"]
    before, after, behind = _media(element)
    before, after, behind = map(material_name, (before, after, behind))
    if label == "Refraction":
        media = f"{before} -> {after}"
    elif label == "Reflection":
        media = f"in {before}, behind {behind}"
    else:
        media = f"in {before}"
    return f"{label} at {surface}, {media}"
