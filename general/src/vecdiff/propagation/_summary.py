"""The printed Solution: one row per element with what its leg could not carry.

Shares are relative amplitudes in percent. A share at or above a tenth of
the 1 % at which a lattice leg refuses (folded, undersampled, band removed)
is flagged with "!", so a result close to refusal is seen before it is.
"""

import textwrap

from vecdiff._printing import (
    count,
    html_table,
    markdown_table,
    number,
    on_plane,
    shorten,
    text_table,
)

# Shares at or above this fraction of a refusal threshold are flagged.
NEAR = 0.1
SHARES = (
    ("unsupported %", "unsupported_relative_amplitude", None),
    ("folded %", "folded_relative_amplitude", "_FOLD_TOLERANCE"),
    ("undersampled %", "undersampled_relative_amplitude", "_SAMPLING_TOLERANCE"),
    ("band removed %", "band_removed_relative_amplitude", "_BAND_TOLERANCE"),
)


def _threshold(name):
    from vecdiff.transport import lattice

    return None if name is None else getattr(lattice, name)


def paired(solution):
    """(index, element, leg, report) for each element; report None where none was made.

    The Propagator reports every element it carries (``carriers`` names the
    leg, None for an identity); a WavePropagator every element between
    unequal indices.
    """
    reports = iter(solution.reports)
    carriers = solution.carriers
    rows = []
    for index, element in enumerate(solution.elements, 1):
        if carriers is not None:
            leg = carriers[index - 1]
            carried = leg is not None
        else:
            leg = solution.legs[index - 1] if index <= len(solution.legs) else None
            carried = element.incident.material.n != element.outgoing.material.n
        rows.append((index, element, leg, next(reports, None) if carried else None))
    return rows


def _element(element):
    from vecdiff._prescription import kind, shape

    label = kind(element)
    if label == "Stop":
        return label, f"r {number(element.radius)}", element.incident.name
    surface = shape(element.surface)["type"]
    if label == "Refraction":
        regions = f"{element.incident.name} -> {element.outgoing.name}"
    elif label == "Reflection":
        regions = f"{element.incident.name}, behind {element.far.name}"
    else:
        regions = element.incident.name
    return label, surface, regions


def _leg(leg):
    if leg is None:
        return "identity"
    name = type(leg).__name__
    if isinstance(leg, tuple):
        return "wave"
    return {"LatticeTransport": "lattice", "MeridianTransport": "meridian"}.get(
        name, name
    )


def _share(report, attribute, threshold):
    value = getattr(report, attribute, None)
    if value is None:
        return ""
    text = number(100 * value, 2)
    if threshold is not None and value >= NEAR * threshold:
        text += "!"
    return text


def table(solution):
    """Headers, alignments and rows of the reports."""
    from vecdiff.transport.report import TransportReport

    headers = ["#", "element", "surface", "regions", "leg"]
    headers += [name for name, _, _ in SHARES]
    headers += ["dark", "caustics", "residual", "orders"]
    aligns = "rllll" + "n" * len(SHARES) + "rrnl"
    rows = []
    for index, element, leg, report in paired(solution):
        row = [str(index), *_element(element), _leg(leg)]
        if isinstance(report, TransportReport):
            row += [_share(report, a, _threshold(t)) for _, a, t in SHARES]
            orders = report.azimuthal_orders
            row += [
                str(report.dark_samples),
                str(report.caustics),
                number(report.newton_residual, 2),
                "(" + ", ".join(str(m) for m in orders) + ")" if orders else "",
            ]
        else:
            row += [""] * (len(SHARES) + 4)
            if leg is None:
                row[4] = "identity (not carried)"
        rows.append(row)
    return headers, aligns, rows


def heading(solution):
    elements, carried = len(solution.elements), len(solution.reports)
    return f"Solution: light through {count(elements, 'element')}, {carried} carried"


def notes(solution):
    from vecdiff.transport.report import TransportReport

    last = solution.spectrum
    lines = [
        f"Leaves as {type(last).__name__} in {last.region.name!r}, "
        f"{on_plane(last.plane)}, {_travel(last)}.",
        "Segments: "
        + ", ".join(
            f"[{i}] {type(s).__name__} in {s.region.name!r}"
            for i, s in enumerate(solution.spectra)
        ),
        "Shares are relative amplitudes in percent; '!' marks one within a decade "
        "of the 1 % at which a lattice leg refuses. residual: the Newton residual "
        "of the inverted map; dark: output samples no ray reaches; caustics: the "
        "caustic index of the carried rays.",
    ]
    others = [r for r in solution.reports if not isinstance(r, TransportReport)]
    lines += [f"report: {r!r}" for r in others]
    return lines


def _travel(spectrum):
    try:
        return "forward (+normal)" if spectrum.direction == 1 else "backward (-normal)"
    except ValueError:
        return "both ways"


def text(solution):
    headers, aligns, rows = table(solution)
    parts = [heading(solution), text_table(headers, rows, aligns)]
    parts += [
        textwrap.fill(line, 100, subsequent_indent="  ") for line in notes(solution)
    ]
    return "\n".join(parts)


def html(solution):
    headers, aligns, rows = table(solution)
    return html_table(
        headers, rows, aligns, caption=heading(solution), notes=notes(solution)
    )


def markdown(solution):
    headers, aligns, rows = table(solution)
    lines = [f"**{heading(solution)}**", "", markdown_table(headers, rows, aligns), ""]
    return "\n".join(lines + [f"- {line}" for line in notes(solution)])


def summary(solution):
    last = solution.spectrum
    return shorten(
        f"Solution({count(len(solution.elements), 'element')}, {len(solution.reports)} "
        f"carried; leaves as {type(last).__name__} in {last.region.name!r}, "
        f"{on_plane(last.plane)})"
    )
