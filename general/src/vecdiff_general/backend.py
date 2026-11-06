"""Where arrays live and compute: NumPy on the CPU or CuPy on a CUDA GPU.

The choice is global, as in diffractsim. Make it once at the top of a script,
before building any vecdiff object:

    from vecdiff_general import backend
    backend.set_backend("CUDA")

Everything built and computed afterwards lives on the GPU. NumPy arrays or
lists passed in are moved to the device on entry; objects built before a switch
keep their arrays on the old device and must be rebuilt. Results stay on the
device until ``to_numpy`` is called; plotting converts on its own. Double
precision throughout: GPU results match the CPU to roundoff. Geometry,
meshing, the independent reference solvers and FINUFFT run on the CPU; their
inputs and outputs are moved for them.

``bd`` is the current array module (``numpy`` or ``cupy``) for user code that
creates arrays on the same device, for example ``backend.bd.linspace``.
"""

from vecdiff_general._backend import bd, get_backend, set_backend, to_numpy

__all__ = ["bd", "get_backend", "set_backend", "to_numpy"]
