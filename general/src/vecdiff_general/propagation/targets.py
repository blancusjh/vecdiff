"""Resolve observation requests to existing placed surfaces and physical grids.

Automatic bounds cover the projected input chart and optical path, not a
claimed energy enclosure or an automatically located focus. Explicit windows
are appropriate for detector crops. Sampling covers the propagating complex
field band, with a margin; it is not a physical accuracy estimate.
"""

import numpy as np

from vecdiff_general.sampling.grids import Grid
from vecdiff_general.surfaces.surface import GraphSurface, Plane


def target_grid(
    source,
    interfaces,
    target=None,
    *,
    x=None,
    y=None,
    z=None,
    window=None,
    shape=None,
    sampling="auto",
):
    coordinates = {k: v for k, v in dict(x=x, y=y, z=z).items() if v is not None}
    if len(coordinates) + (target is not None) != 1:
        raise ValueError("supply one target surface/grid or exactly one of x=, y=, z=")
    if sampling != "auto":
        raise ValueError(
            "sampling must be 'auto'; specify shape or an explicit sampled surface to override"
        )
    if coordinates:
        name, value = next(iter(coordinates.items()))
        target = getattr(Plane, name)(value)
    if isinstance(target, Grid):
        if window is not None or shape is not None:
            raise ValueError("a sampled target already defines its window and shape")
        return target, {"mode": "explicit", "shape": target.shape}
    if not isinstance(target, GraphSurface):
        raise TypeError("target must be a placed surface or a physical grid")

    plane = source.grid
    width = max(len(plane.u) * plane.spacing[0], len(plane.v) * plane.spacing[1])
    if window is None:
        # Project a finite source/path box onto the requested chart. Include
        # one input-chart width downstream for longitudinal observation planes.
        positions = [plane.points.reshape(-1, 3)]
        for interface in interfaces:
            xy = np.stack(np.meshgrid(plane.u, plane.v), axis=-1)
            positions.append(interface.surface.points(xy).reshape(-1, 3))
        end = interfaces[-1].surface.frame.origin if interfaces else plane.frame.origin
        positions.append(
            (
                plane.points + end - plane.frame.origin + width * plane.frame.normal
            ).reshape(-1, 3)
        )
        points = np.concatenate(positions)
        uv = (points - target.frame.origin) @ target.frame.basis[:, :2]
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        window = (lo[0], hi[0], lo[1], hi[1])
    bounds = np.asarray(window, dtype=float)
    if (
        bounds.shape != (4,)
        or not np.isfinite(bounds).all()
        or bounds[1] <= bounds[0]
        or bounds[3] <= bounds[2]
    ):
        raise ValueError(
            "window must be (umin, umax, vmin, vmax) with increasing finite bounds"
        )
    automatic_shape = shape is None
    if shape is None:
        max_index = max(
            [source.expansion.region.material.n]
            + [i.plus.material.n for i in interfaces]
        )
        step = 0.4 * source.expansion.wavelength0 / max_index
        # A curved chart stretches physical distance; bound using a pilot grid.
        u, v = np.meshgrid(
            np.linspace(bounds[0], bounds[1], 17), np.linspace(bounds[2], bounds[3], 17)
        )
        gradient = target.gradient(np.stack((u, v), axis=-1))
        stretch = np.sqrt(1 + np.max(gradient**2, axis=(0, 1)))
        lengths = np.array([bounds[1] - bounds[0], bounds[3] - bounds[2]])
        counts = np.maximum(2, np.ceil(lengths * stretch / step).astype(int) + 1)
        if np.prod(counts.astype(float)) > 2_000_000:
            raise ValueError(
                "automatic target exceeds two million samples; request a smaller window or an explicit shape"
            )
        shape = tuple(int(k) for k in counts[::-1])
    if len(shape) != 2 or any(
        not isinstance(k, (int, np.integer)) or k < 2 for k in shape
    ):
        raise ValueError("shape must be (Nv, Nu), with at least two samples per axis")
    u = np.linspace(bounds[0], bounds[1], shape[1])
    v = np.linspace(bounds[2], bounds[3], shape[0])
    grid = target.sample(u=u, v=v)
    return grid, dict(
        mode="automatic" if automatic_shape else "surface",
        window=tuple(bounds),
        shape=grid.shape,
        spacing=(float(u[1] - u[0]), float(v[1] - v[0])),
        bounds_policy="projected input chart and path unless explicitly supplied",
    )
