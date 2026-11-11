"""The printed form of vecdiff objects: numbers, vectors, aligned blocks and tables.

Every object prints through these helpers, so the style is one: numbers to
six significant digits without trailing noise and with a compact exponent
(``5.32e-7``), vectors as tuples, ``name: value`` blocks aligned on the
colon, and tables as plain monospace text or as HTML for Jupyter. A repr is
one line that never forms a field nor reads more than a few numbers of an
array; ``guarded`` makes every printing method total, so printing never
raises.
"""

import functools
import html
import math

from vecdiff._backend import to_numpy

DIGITS = 6
REPR_LIMIT = 200
# Components smaller than this share of a vector's largest are roundoff.
SNAP = 1e-12


def guarded(method):
    """A printing method that falls back to a plain description instead of raising."""

    @functools.wraps(method)
    def printing(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as error:  # noqa: BLE001 - printing must not raise
            if method.__name__ == "_repr_html_":
                return None
            return f"<{type(self).__name__} (unprintable: {type(error).__name__})>"

    return printing


def number(value, digits=DIGITS):
    """A real or complex number: six significant digits, compact exponent."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    value = to_numpy(value)
    if getattr(value, "size", 1) != 1:
        return f"<{getattr(value, 'shape', '?')} array>"
    value = value.item() if hasattr(value, "item") else value
    if isinstance(value, complex):
        return _complex(value, digits)
    return _real(float(value), digits)


def _real(value, digits):
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == 0:
        return "0"
    text = f"{value:.{digits}g}"
    mantissa, _, exponent = text.partition("e")
    return f"{mantissa}e{int(exponent)}" if exponent else mantissa


def _complex(value, digits):
    re, im = value.real, value.imag
    if im == 0:
        return _real(re, digits)
    imaginary = f"{_real(abs(im), digits)}j"
    if re == 0:
        return imaginary if im > 0 else f"-{imaginary}"
    return f"{_real(re, digits)}{'+' if im > 0 else '-'}{imaginary}"


def values(array, limit=6):
    """Up to ``limit`` numbers of a small array as a flat host list, else None."""
    array = to_numpy(array)
    if array.size > limit:
        return None
    return array.ravel().tolist()


def vector(array, digits=DIGITS):
    """A short vector as a tuple, roundoff components shown as 0: (0, 0, 0.003)."""
    items = values(array)
    if items is None:
        return f"<{to_numpy(array).shape} array>"
    scale = max((abs(x) for x in items), default=0.0)
    items = [0.0 if abs(x) <= SNAP * scale else x for x in items]
    return "(" + ", ".join(number(x, digits) for x in items) + ")"


def is_vector(array, expected, tolerance=SNAP):
    """Whether a 3-vector equals ``expected`` to roundoff."""
    items = values(array)
    return items is not None and all(
        abs(a - b) <= tolerance for a, b in zip(items, expected)
    )


def span(axis):
    """The first and last values of a 1D axis, read without forming more."""
    return number(axis[0]), number(axis[-1])


def shape(dims):
    """A shape as 283 x 283."""
    return " x ".join(str(int(n)) for n in dims) if len(dims) else "scalar"


def name_of(thing):
    """The name a callable or object goes by, for a profile or a function."""
    if isinstance(thing, (functools.partial,)):
        return f"partial({name_of(thing.func)})"
    qualname = getattr(thing, "__qualname__", None) or type(thing).__qualname__
    return "lambda" if qualname.endswith("<lambda>") else qualname.split(".")[-1]


def call(name, *parts, **keywords):
    """``name(part, key=value, ...)``, skipping None keywords; shortened to fit a line."""
    items = [str(p) for p in parts if p is not None]
    items += [f"{key}={value}" for key, value in keywords.items() if value is not None]
    return shorten(f"{name}({', '.join(items)})")


def shorten(text, limit=REPR_LIMIT):
    """One line of at most ``limit`` characters."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def block(title, pairs, indent="  "):
    """A title over ``name: value`` lines aligned on the colon; None values are left out.

    A value with several lines continues under its first line.
    """
    pairs = [(str(k), str(v)) for k, v in pairs if v is not None]
    width = max((len(k) for k, _ in pairs), default=0)
    lines = [title] if title else []
    for key, value in pairs:
        first, *rest = value.split("\n")
        lines.append(f"{indent}{key + ':':<{width + 1}} {first}")
        lines += [f"{indent}{'':<{width + 1}} {line}" for line in rest]
    return "\n".join(lines)


def indented(text, indent="  "):
    return "\n".join(indent + line if line else line for line in str(text).split("\n"))


# --- Tables -------------------------------------------------------------


def _split(cell):
    """A numeric cell around its decimal point (or exponent, or end)."""
    for mark in (".", "e"):
        if mark in cell:
            index = cell.index(mark)
            return cell[:index], cell[index:]
    return cell, ""


def _column(cells, align):
    """A column's cells padded to one width: left, right or decimal-aligned."""
    if align == "n":
        parts = [_split(c) if _numeric(c) else None for c in cells]
        left = max((len(p[0]) for p in parts if p), default=0)
        right = max((len(p[1]) for p in parts if p), default=0)
        cells = [
            p[0].rjust(left) + p[1].ljust(right) if p else c
            for c, p in zip(cells, parts)
        ]
    width = max((len(c) for c in cells), default=0)
    pad = str.rjust if align in "rn" else str.ljust
    return [pad(c, width) for c in cells]


def _numeric(cell):
    try:
        float(cell)
    except ValueError:
        return False
    return True


def text_table(headers, rows, aligns=None, gap="  "):
    """Rows of cells under their headers, columns aligned, a rule under the header.

    ``aligns`` holds one of "l" (left), "r" (right) or "n" (numbers aligned on
    their decimal point) per column.
    """
    aligns = aligns or "l" * len(headers)
    columns = [
        _column([h] + [row[i] for row in rows], a)
        for i, (h, a) in enumerate(zip(headers, aligns))
    ]
    # Headers of number columns sit right, over the digits.
    for column, a, h in zip(columns, aligns, headers):
        if a in "nr":
            column[0] = h.rjust(len(column[0]))
    lines = [gap.join(cells).rstrip() for cells in zip(*columns)]
    rule = gap.join("-" * len(column[0]) for column in columns)
    return "\n".join([lines[0], rule, *lines[1:]])


def html_table(headers, rows, aligns=None, caption=None, notes=()):
    """The same table as HTML, numbers right-aligned, with a caption and notes."""
    aligns = aligns or "l" * len(headers)
    style = ' style="text-align:right"'

    def cell(tag, text, align):
        return f"<{tag}{style if align in 'nr' else ''}>{html.escape(text)}</{tag}>"

    head = "".join(cell("th", h, a) for h, a in zip(headers, aligns))
    body = "".join(
        "<tr>" + "".join(cell("td", c, a) for c, a in zip(row, aligns)) + "</tr>"
        for row in rows
    )
    title = f"<caption>{html.escape(caption)}</caption>" if caption else ""
    remarks = "".join(f"<p>{html.escape(note)}</p>" for note in notes)
    return (
        f"<div><table>{title}<thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table>{remarks}</div>"
    )


def markdown_table(headers, rows, aligns=None):
    """The same table in Markdown."""
    aligns = aligns or "l" * len(headers)

    def escape(text):
        return text.replace("|", "\\|")

    lines = ["| " + " | ".join(escape(h) for h in headers) + " |"]
    lines.append("|" + "|".join("---:" if a in "nr" else ":---" for a in aligns) + "|")
    lines += ["| " + " | ".join(escape(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)
