"""Visualization utilities for radial scalar fields and field comparisons."""

import numpy as np
import matplotlib.pyplot as plt

from .coordinate_transformation import polar_grid_to_cartesian_grid


def _radial_mesh(half_size, n_points):
    x = np.linspace(-half_size, half_size, n_points)
    y = np.linspace(-half_size, half_size, n_points)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    extent = [-half_size, half_size, -half_size, half_size]
    return rr, extent


def radial_map(values, radial_axis, half_size, n_img):
    """Build a 2D scalar map from a 1D radial profile."""
    radial_axis = np.asarray(radial_axis)
    values = np.asarray(values)
    if radial_axis.ndim != 1:
        raise ValueError("`radial_axis` must be 1D.")
    if values.ndim != 1:
        raise ValueError("`values` must be 1D.")
    if len(values) != len(radial_axis):
        raise ValueError("`values` and `radial_axis` must have the same length.")

    rr, extent = _radial_mesh(half_size, n_img)
    img = np.interp(rr, radial_axis, values, left=values[0], right=values[-1])
    return img, extent


def _component_labels(initial_field):
    if hasattr(initial_field, "L") and hasattr(initial_field, "R"):
        return ("L", "R"), ("Input Circular Components", "Transmitted Circular Components")
    if hasattr(initial_field, "x") and hasattr(initial_field, "y"):
        return ("x", "y"), ("Input Cartesian Components", "Transmitted Cartesian Components")
    if hasattr(initial_field, "r") and hasattr(initial_field, "phi"):
        return ("r", r"\varphi"), ("Input Polar Components", "Transmitted Polar Components")
    return ("1", "2"), ("Input Components", "Transmitted Components")


def _labels_from_representation(representation):
    rep = str(representation).lower()
    if rep == "circular":
        return ("L", "R"), ("Input Circular Components", "Propagated Circular Components")
    if rep == "polar":
        return ("r", r"\varphi"), ("Input Polar Components", "Propagated Polar Components")
    if rep == "cartesian":
        return ("x", "y"), ("Input Cartesian Components", "Propagated Cartesian Components")
    raise ValueError("`representation` must be one of: 'circular', 'polar', 'cartesian'.")


def _extract_input_components(initial_field):
    if hasattr(initial_field, "L") and hasattr(initial_field, "R"):
        return np.asarray(initial_field.L), np.asarray(initial_field.R)
    if hasattr(initial_field, "x") and hasattr(initial_field, "y"):
        return np.asarray(initial_field.x), np.asarray(initial_field.y)
    if hasattr(initial_field, "r") and hasattr(initial_field, "phi"):
        return np.asarray(initial_field.r), np.asarray(initial_field.phi)
    raise ValueError("Could not extract input components from `initial_field`.")


def _extract_transmitted_components(transmitted):
    if hasattr(transmitted, "L") and hasattr(transmitted, "R"):
        return np.asarray(transmitted.L), np.asarray(transmitted.R)
    if hasattr(transmitted, "r") and hasattr(transmitted, "phi"):
        return np.asarray(transmitted.r), np.asarray(transmitted.phi)
    if hasattr(transmitted, "x") and hasattr(transmitted, "y"):
        return np.asarray(transmitted.x), np.asarray(transmitted.y)

    if isinstance(transmitted, dict):
        keys = ("L", "R") if "L" in transmitted and "R" in transmitted else ("x", "y")
        if keys[0] in transmitted and keys[1] in transmitted:
            return np.asarray(transmitted[keys[0]]), np.asarray(transmitted[keys[1]])
        if "E1" in transmitted and "E2" in transmitted:
            return np.asarray(transmitted["E1"]), np.asarray(transmitted["E2"])
        raise ValueError(
            "When `transmitted` is a dict, use keys ('L','R'), ('x','y'), or ('E1','E2')."
        )

    if isinstance(transmitted, (tuple, list)) and len(transmitted) == 2:
        return np.asarray(transmitted[0]), np.asarray(transmitted[1])

    if isinstance(transmitted, np.ndarray) and transmitted.ndim >= 1 and transmitted.shape[0] == 2:
        return np.asarray(transmitted[0]), np.asarray(transmitted[1])

    raise ValueError(
        "`transmitted` must be a tuple/list (component1, component2), dict, or ndarray with shape[0]==2."
    )


def field_cartesian_maps(field, half_size=None, n_img=500):
    """Sample a polar-grid Field on a Cartesian plotting mesh."""
    if field.grid.type != "polar":
        raise ValueError("`field` must be sampled on a polar grid.")

    component1, component2 = _extract_input_components(field)
    labels, _ = _component_labels(field)
    radial_axis = np.asarray(field.grid.r)
    varphi = np.asarray(field.grid.varphi)
    if half_size is None:
        half_size = float(np.max(radial_axis))

    rr, extent = _radial_mesh(half_size, n_img)
    if component1.ndim == 1:
        map1 = np.interp(rr, radial_axis, component1, left=component1[0], right=0.0)
        map2 = np.interp(rr, radial_axis, component2, left=component2[0], right=0.0)
        return map1, map2, extent, labels

    x = np.linspace(-half_size, half_size, n_img)
    xx, yy = np.meshgrid(x, x)
    map1 = polar_grid_to_cartesian_grid(component1, radial_axis, varphi, xx, yy, fill_value=0.0)
    map2 = polar_grid_to_cartesian_grid(component2, radial_axis, varphi, xx, yy, fill_value=0.0)
    return map1, map2, extent, labels


def plot_field(field, half_size=None, n_img=500, cmap="hot", component_view="abs", title=None):
    """Plot a Field through its native component representation."""
    component1, component2, extent, labels = field_cartesian_maps(field, half_size, n_img)
    return plot_field_2d_components(
        component1,
        component2,
        extent,
        labels=labels,
        cmap=cmap,
        component_view=component_view,
        title=title,
    )

def plot_radial_field(
    f,
    L=10,
    N=500,
    cmap="hot",
    title=None,
    vmin = None,
    vmax = None
):
    """
    Grafica un campo escalar radial 2D definido por f(r).

    Parámetros
    ----------
    f : callable
        Función radial f(r).
    L : float
        Mitad del tamaño del dominio:
        x,y ∈ [-L, L]
    N : int
        Número de puntos por eje.
    cmap : str
        Colormap de matplotlib.
    title : str
        Título de la figura.
    """

    # Malla cartesiana y coordenada radial
    R, extent = _radial_mesh(L, N)

    # Evaluación del campo
    F = f(R)


    
    # Gráfica
    plt.figure(figsize=(7,7))

    plt.imshow(
        F,
        extent=extent,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")

    if title is not None:
        plt.title(title)

    
    plt.colorbar(label=r"$f(r)$")
    plt.gca().set_aspect("equal")
    plt.show()


def plot_results(
    initial_field,
    propagated_field,
    radial_axis,
    input_radial_axis=None,
    half_size=None,
    input_half_size=None,
    propagated_half_size=None,
    n_img=500,
    cmap="hot",
    show=True,
    params=None,
    representation=None,
    component_view="abs",
    **kwargs,
):
    """
    Plot input and propagated scalar maps by components and intensity.

    The figure has 2 rows x 3 columns:
    - Row 1: input component 1, input component 2, input intensity
    - Row 2: propagated component 1, propagated component 2, propagated intensity

    Parameters metadata can be provided either via `params` dict or kwargs
    (e.g. n0, z0, ni, zi, wavelength/lambda).
    """
    radial_axis = np.asarray(radial_axis)
    if radial_axis.ndim != 1:
        raise ValueError("`radial_axis` must be 1D.")
    if input_radial_axis is None:
        input_radial_axis = radial_axis
    input_radial_axis = np.asarray(input_radial_axis)
    if input_radial_axis.ndim != 1:
        raise ValueError("`input_radial_axis` must be 1D.")
    if half_size is None:
        half_size = float(np.max(radial_axis))
    if input_half_size is None:
        input_half_size = half_size if propagated_half_size is None else float(np.max(input_radial_axis))
    if propagated_half_size is None:
        propagated_half_size = half_size

    in1, in2 = _extract_input_components(initial_field)
    out1, out2 = _extract_transmitted_components(propagated_field)

    if not (len(out1) == len(out2) == len(radial_axis)):
        raise ValueError("Propagated components and `radial_axis` must have the same length.")
    if not (len(in1) == len(in2) == len(input_radial_axis)):
        raise ValueError("Input components and `input_radial_axis` must have the same length.")

    if representation is None:
        labels, row_titles = _component_labels(initial_field)
        row_titles = (row_titles[0], row_titles[1].replace("Transmitted", "Propagated"))
        coord_name = "auto"
    else:
        labels, row_titles = _labels_from_representation(representation)
        coord_name = str(representation).lower()

    if component_view == "abs":
        rep = np.abs
    elif component_view == "real":
        rep = np.real
    elif component_view == "imag":
        rep = np.imag
    else:
        raise ValueError("`component_view` must be one of: 'abs', 'real', 'imag'.")

    amp_in1 = rep(in1)
    amp_in2 = rep(in2)
    amp_out1 = rep(out1)
    amp_out2 = rep(out2)

    intensity_in = amp_in1**2 + amp_in2**2
    intensity_out = amp_out1**2 + amp_out2**2

    img_in1, extent_in = radial_map(amp_in1, input_radial_axis, input_half_size, n_img)
    img_in2, _ = radial_map(amp_in2, input_radial_axis, input_half_size, n_img)
    img_i_in, _ = radial_map(intensity_in, input_radial_axis, input_half_size, n_img)
    img_out1, extent_out = radial_map(amp_out1, radial_axis, propagated_half_size, n_img)
    img_out2, _ = radial_map(amp_out2, radial_axis, propagated_half_size, n_img)
    img_i_out, _ = radial_map(intensity_out, radial_axis, propagated_half_size, n_img)

    images = [img_in1, img_in2, img_i_in, img_out1, img_out2, img_i_out]
    extents = [extent_in, extent_in, extent_in, extent_out, extent_out, extent_out]

    titles = [
        rf"{row_titles[0]} ({coord_name}): $E_{labels[0]}$",
        rf"{row_titles[0]} ({coord_name}): $E_{labels[1]}$",
        rf"{row_titles[0]} Intensity: $|E_{labels[0]}|^2 + |E_{labels[1]}|^2$",
        rf"{row_titles[1]} ({coord_name}): $E'_{labels[0]}$",
        rf"{row_titles[1]} ({coord_name}): $E'_{labels[1]}$",
        rf"{row_titles[1]} Intensity: $|E'_{labels[0]}|^2 + |E'_{labels[1]}|^2$",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, img, title, extent in zip(axes.flat, images, titles, extents):
        img = np.abs(np.asarray(img))
        vmax = float(np.max(img))
        if vmax == 0.0:
            vmax = 1.0
        im = ax.imshow(img, extent=extent, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    metadata = {}
    if params is not None:
        metadata.update(params)
    metadata.update(kwargs)

    if metadata:
        ordered_keys = ["n0", "z0", "ni", "zi", "lambda", "wavelength", "R", "N"]
        shown = []
        for k in ordered_keys:
            if k in metadata:
                shown.append(f"{k}={metadata[k]}")
        for k, v in metadata.items():
            if k not in ordered_keys:
                shown.append(f"{k}={v}")
        fig.suptitle("Field Results", fontsize=14)
        fig.text(0.01, 0.01, " | ".join(shown), ha="left", va="bottom", fontsize=10)

    if show:
        plt.show()
    return fig, axes


def plot_transmission_comparison(*args, **kwargs):
    """Backward-compatible alias for plot_results."""
    return plot_results(*args, **kwargs)


def plot_field_2d_components(
    component1,
    component2,
    extent,
    labels=("1", "2"),
    cmap="hot",
    component_view="abs",
    title=None,
    field_symbol="E'",
):
    c1 = np.asarray(component1)
    c2 = np.asarray(component2)

    if component_view == "abs":
        rep = np.abs
    elif component_view == "real":
        rep = np.real
    elif component_view == "imag":
        rep = np.imag
    else:
        raise ValueError("`component_view` must be one of: 'abs', 'real', 'imag'.")

    i1 = rep(c1)
    i2 = rep(c2)
    intensity = i1**2 + i2**2

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    images = [i1, i2, intensity]
    titles = [
        rf"${field_symbol}_{labels[0]}$",
        rf"${field_symbol}_{labels[1]}$",
        rf"$|{field_symbol}_{labels[0]}|^2 + |{field_symbol}_{labels[1]}|^2$",
    ]

    for ax, img, ttl in zip(axes, images, titles):
        vmax = float(np.max(np.abs(img)))
        if vmax == 0.0:
            vmax = 1.0
        im = ax.imshow(np.abs(img), extent=extent, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(ttl)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title is not None:
        fig.suptitle(title)
    plt.show()
    return fig, axes


def plot_field_2d_intensity(component1, component2, extent, cmap="hot", component_view="abs", title=None):
    c1 = np.asarray(component1)
    c2 = np.asarray(component2)

    if component_view == "abs":
        rep = np.abs
    elif component_view == "real":
        rep = np.real
    elif component_view == "imag":
        rep = np.imag
    else:
        raise ValueError("`component_view` must be one of: 'abs', 'real', 'imag'.")

    i1 = rep(c1)
    i2 = rep(c2)
    intensity = i1**2 + i2**2

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.8), constrained_layout=True)
    vmax = float(np.max(np.abs(intensity)))
    if vmax == 0.0:
        vmax = 1.0
    im = ax.imshow(np.abs(intensity), extent=extent, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_title(r"$|E_1|^2 + |E_2|^2$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title is not None:
        fig.suptitle(title)
    plt.show()
    return fig, ax
