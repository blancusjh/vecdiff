"""Vector optics by eikonal-angular-spectrum transport.

Public objects are reached through the namespace of what they are:
``media``, ``geometry`` and ``surfaces`` describe space and matter, ``system``
the optical system, ``fields`` the light, ``transport`` its transport across
each surface, ``propagation`` its propagation through the whole system and
``observables`` what is measured. ``alternatives`` holds other methods.
Nothing is re-exported at this level.
``backend.set_backend("CUDA")`` moves arrays and computation to a GPU.
Plotting (``visualization``) and the exact solutions used as references
(``references``) are imported explicitly when needed.
"""

from vecdiff import (
    alternatives,
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
    transport,
    volumes,
)

__all__ = [
    "alternatives",
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
    "transport",
    "volumes",
]
