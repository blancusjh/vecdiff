"""Vector optics: fields, physical systems, transport, interaction and pipelines.

Public objects are reached through the namespace of what they are, for
example ``surfaces.SphericalCap``, ``fields.angular_spectrum`` or
``propagation.PropagationPipeline``. Nothing is re-exported at this level.
``backend.set_backend("CUDA")`` moves arrays and computation to a GPU.
Plotting (``visualization``) and independent references
(``propagation.references``) are imported explicitly when needed.
"""

from vecdiff_general import (
    backend,
    catalog,
    fields,
    fourier,
    geometry,
    io,
    media,
    observables,
    propagation,
    sampling,
    surfaces,
    system,
    volumes,
)

__all__ = [
    "backend",
    "catalog",
    "fields",
    "fourier",
    "geometry",
    "io",
    "media",
    "observables",
    "propagation",
    "sampling",
    "surfaces",
    "system",
    "volumes",
]
