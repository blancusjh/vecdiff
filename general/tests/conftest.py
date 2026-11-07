"""Run the whole suite on either backend: ``VECDIFF_BACKEND=CUDA pytest general/tests``.

Under CUDA the NumPy comparison helpers accept device arrays by copying them
to the host first; everything else in the tests runs unchanged on the GPU.
"""

import os

import numpy.testing
import vecdiff_general as o

if os.environ.get("VECDIFF_BACKEND", "CPU") == "CUDA":
    o.backend.set_backend("CUDA")

    def _host(value):
        if isinstance(value, (list, tuple)):
            return type(value)(_host(item) for item in value)
        return o.backend.to_numpy(value) if hasattr(value, "get") else value

    for _name in (
        "assert_allclose",
        "assert_array_equal",
        "assert_array_less",
        "assert_equal",
        "assert_array_almost_equal",
    ):

        def _compare(*args, _original=getattr(numpy.testing, _name), **kwargs):
            return _original(*(_host(arg) for arg in args), **kwargs)

        setattr(numpy.testing, _name, _compare)
