"""Single-axis intensity rendering with optional polarization."""

import numpy as np

from .polarization import (
    _colorbar,
    _draw_values,
    _plane_data,
    _prepare_axis,
    _view_limits,
    plot_polarization,
)


def plot_intensity(
    field,
    intensity=None,
    *,
    axis=None,
    norm=None,
    cmap="inferno",
    label=r"$|\mathbf{E}|^2$ (V²/m²)",
    title=None,
    weights=None,
    polarization=True,
    **glyph_options,
):
    """Draw a physical intensity map, optionally with projected polarization.

    Default background is total |E|², including normal E. A supplied grid-shaped
    intensity can instead show a component, flux, or scalar reference; label must
    state its meaning. Glyphs always describe the supplied field's u/v projection.
    Ensembles use weighted |E|² and polarized-component glyphs (opacity = DoP).
    Use polarization=False for component or scalar-reference panels, or when
    the overlay adds no information to an intensity comparison.
    Canvas arrangement and shared color normalization belong to the caller.
    """
    from matplotlib.colors import Normalize

    members = field if isinstance(field, (tuple, list)) else [field]

    if not members:
        raise ValueError("an intensity map needs at least one field")

    members = tuple(_plane_data(member, 0)[0] for member in members)
    electric, u, v, *_ = _plane_data(members[0], 0)

    if intensity is None:
        from ..observables.polarization import ensemble_weights

        coefficients = ensemble_weights(members, weights)
        intensity = sum(c * member.norm2() for c, member in zip(coefficients, members))

    intensity = np.asarray(intensity)

    if intensity.shape != electric.grid.shape or not np.isfinite(intensity).all():
        raise ValueError("intensity must be finite and have the field grid shape")

    glyph_options.setdefault("crop", False)
    glyph_options.setdefault("legend", False)
    if polarization:
        axis = plot_polarization(
            field,
            axis=axis,
            background=False,
            colorbar=False,
            title=title,
            weights=weights,
            **glyph_options,
        )
    else:
        limits = _view_limits(u, v, intensity > 0, glyph_options["crop"])
        axis = _prepare_axis(axis, limits, glyph_options.get("figsize"), title)

    image = _draw_values(
        axis,
        electric.grid,
        u,
        v,
        intensity,
        cmap=cmap,
        norm=norm or Normalize(0, intensity.max() or 1),
        smooth=True,
    )
    image.set_zorder(0)
    _colorbar(axis, image, label)

    return axis
