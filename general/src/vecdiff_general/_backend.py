"""Array backend state and the few operations whose call differs between devices.

``bd`` stands for the current array module, NumPy on the CPU or CuPy on a
CUDA GPU, as in diffractsim. Library code writes ``bd.exp``, ``bd.stack`` and
so on; transforms and special functions with device-specific implementations
are the functions below. The public switch is ``vecdiff_general.backend``.
"""

import functools

import numpy as np

_state = {"name": "CPU", "module": np}


def _cupy():
    try:
        import cupy
    except ImportError as error:
        raise RuntimeError(
            "the CUDA backend needs CuPy built for the installed CUDA toolkit; "
            "install the cuda12 or cuda13 extra"
        ) from error

    try:
        cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as error:
        raise RuntimeError("CuPy is installed but no CUDA device is usable") from error

    return cupy


def is_device_array(array):
    """True for a CuPy array, without importing CuPy."""
    return type(array).__module__.split(".")[0] == "cupy"


def to_numpy(array):
    """NumPy copy of a CuPy array; NumPy arrays and scalars pass through."""
    if is_device_array(array):
        return array.get()

    return np.asarray(array)


class _ArrayModule:
    """The current array module; arrays from the other device are moved on entry."""

    def __getattr__(self, name):
        return getattr(_state["module"], name)

    def __repr__(self):
        return f"<array module {_state['module'].__name__}>"

    def asarray(self, value, dtype=None):
        if _state["name"] == "CPU":
            return np.asarray(_host_nested(value), dtype=dtype)

        return _state["module"].asarray(_device_nested(value), dtype=dtype)

    def array(self, value, dtype=None, copy=True):
        if _state["name"] == "CPU":
            return np.array(_host_nested(value), dtype=dtype, copy=copy)

        return _state["module"].array(_device_nested(value), dtype=dtype, copy=copy)


def _holds_device_array(value):
    if isinstance(value, (list, tuple)):
        return any(_holds_device_array(item) for item in value)
    return is_device_array(value)


def _host_nested(value):
    """Lists and tuples of arrays from either device, as NumPy can read them."""
    if isinstance(value, (list, tuple)) and _holds_device_array(value):
        return [_host_nested(item) for item in value]
    return to_numpy(value) if is_device_array(value) else value


def _device_nested(value):
    """CuPy builds arrays from nested sequences only when they hold no CuPy arrays."""
    if isinstance(value, (list, tuple)) and _holds_device_array(value):
        cupy = _state["module"]
        return cupy.stack([cupy.asarray(_device_nested(item)) for item in value])
    return value


bd = _ArrayModule()


def fft2(array, axes=(0, 1), workers=-1, overwrite_x=False):
    if _state["name"] == "CPU":
        import scipy.fft

        return scipy.fft.fft2(
            array, axes=axes, workers=workers, overwrite_x=overwrite_x
        )

    return _state["module"].fft.fft2(array, axes=axes)


def ifft2(array, axes=(0, 1), workers=-1, overwrite_x=False):
    if _state["name"] == "CPU":
        import scipy.fft

        return scipy.fft.ifft2(
            array, axes=axes, workers=workers, overwrite_x=overwrite_x
        )

    return _state["module"].fft.ifft2(array, axes=axes)


def fft(array, axis=-1):
    if _state["name"] == "CPU":
        import scipy.fft

        return scipy.fft.fft(array, axis=axis, workers=-1)

    return _state["module"].fft.fft(array, axis=axis)


def ifft(array, axis=-1):
    if _state["name"] == "CPU":
        import scipy.fft

        return scipy.fft.ifft(array, axis=axis, workers=-1)

    return _state["module"].fft.ifft(array, axis=axis)


def host_operator(shape, matvec, rmatvec, dtype):
    """SciPy LinearOperator whose products run on the current device.

    Iterative solvers stay SciPy's on either backend: CuPy's lsmr casts complex
    vectors to real. Only the solver's vectors cross between host and device.
    """
    import scipy.sparse.linalg

    if _state["name"] != "CPU":
        device_matvec, device_rmatvec = matvec, rmatvec

        def matvec(vector):
            return to_numpy(device_matvec(bd.asarray(vector)))

        def rmatvec(vector):
            return to_numpy(device_rmatvec(bd.asarray(vector)))

    return scipy.sparse.linalg.LinearOperator(
        shape, matvec=matvec, rmatvec=rmatvec, dtype=dtype
    )


def divide(numerator, denominator, where):
    """numerator / denominator where `where` holds and zero elsewhere.

    CuPy's ufuncs take no ``where=``; this form is the same on both devices.
    """
    numerator, denominator = bd.asarray(numerator), bd.asarray(denominator)
    quotient = numerator / bd.where(where, denominator, 1)

    return bd.where(where, quotient, bd.zeros_like(quotient))


def _legendre(count, theta):
    """P_count(cos theta) and dP_count/dtheta for 0 < theta <= pi / 2.

    The recurrence runs on the differences D_j = P_j - P_(j-1) with
    y = 1 - x = 2 sin²(theta / 2), j D_j = (j - 1) D_(j-1) - (2j - 1) y P_(j-1),
    so a node near x = 1 is not limited by the rounding of cos theta.
    """
    y = 2 * np.sin(theta / 2) ** 2
    value, difference = 1 - y, -y
    for j in range(2, count + 1):
        difference = ((j - 1) * difference - (2 * j - 1) * y * value) / j
        value = value + difference
    return value, count * (difference - y * value) / np.sin(theta)


@functools.lru_cache(maxsize=64)
def _gauss_legendre(count):
    """Nodes x = cos theta by Newton steps in theta from Tricomi's estimates.

    The weight is 2 / (dP/dtheta)²: in theta neither 1 - x² nor P' cancels
    near the ends, where the companion-matrix rule loses 6e-7 of the weight
    at 4000 nodes. Cost O(count²), against O(count³) for the eigenvalues.
    """
    if count == 1:
        return np.zeros(1), np.full(1, 2.0)
    k = np.arange(1, (count + 1) // 2 + 1)
    theta = np.pi * (4 * k - 1) / (4 * count + 2)
    theta = theta + (1 / (8 * count**2) - 1 / (8 * count**3)) / np.tan(theta)
    for _ in range(100):
        value, slope = _legendre(count, theta)
        step = value / slope
        theta = theta - step
        if np.max(np.abs(step)) < 4 * np.finfo(float).eps:
            break
    value, slope = _legendre(count, theta)
    x, w = np.cos(theta), 2 / slope**2
    if count % 2:
        x[-1] = 0.0
        return np.r_[-x, x[-2::-1]], np.r_[w, w[-2::-1]]
    return np.r_[-x, x[::-1]], np.r_[w, w[::-1]]


def leggauss(count):
    """Gauss-Legendre nodes (ascending) and weights on [-1, 1], on the current device."""
    count = int(count)
    if count < 1:
        raise ValueError("the Gauss-Legendre rule needs at least one node")
    nodes, weights = _gauss_legendre(count)

    return bd.asarray(nodes.copy()), bd.asarray(weights.copy())


def gammaln(x):
    if _state["name"] == "CPU":
        from scipy.special import gammaln as host

        return host(x)

    from cupyx.scipy.special import gammaln as device

    return device(x)


# J_n(x) for integer n and real x on the GPU. Seeds J0, J1 are Cephes (the
# routines SciPy uses, bundled with CuPy); forward recurrence is stable for
# x >= n, Miller's downward recurrence normalized by J0 + 2 sum J_2k = 1 serves
# x < n. Against 30-digit references the error stays below 4e-15 of the
# envelope max(|J|, x^-1/2); SciPy's jv drifts to 1e-12 near x = 3000.
_BESSEL_J = r"""
int m = n < 0 ? -n : n;
double sign = (n < 0 && (m & 1)) ? -1.0 : 1.0;
double ax = x;
if (ax < 0) { ax = -ax; if (m & 1) sign = -sign; }
double r;
if (ax == 0.0) {
    r = (m == 0) ? 1.0 : 0.0;
} else if (ax >= m) {
    double a = xsf::cephes::j0(ax), b = xsf::cephes::j1(ax);
    if (m == 0) {
        r = a;
    } else {
        for (int k = 1; k < m; ++k) { double c = (2.0 * k / ax) * b - a; a = b; b = c; }
        r = b;
    }
} else {
    int start = m + (int)ax + 20 + (int)sqrt(50.0 * (m + 1));
    start += start & 1;
    double up = 0.0, cur = 1e-280, sum = 0.0, keep = 0.0;
    for (int k = start; k > 0; --k) {
        double down = (2.0 * k / ax) * cur - up;
        up = cur;
        cur = down;
        if (k - 1 == m) keep = cur;
        if (k - 1 > 0 && ((k - 1) & 1) == 0) sum += 2.0 * cur;
        if (fabs(cur) > 1e200) { cur *= 1e-200; up *= 1e-200; sum *= 1e-200; keep *= 1e-200; }
    }
    r = keep / (sum + cur);
}
y = sign * r;
"""
_kernels = {}


def jv(order, x):
    """Bessel function of the first kind for integer orders and real arguments."""
    if _state["name"] == "CPU":
        from scipy.special import jv as host

        return host(order, x)

    cupy = _state["module"]
    order_host = to_numpy(order)

    if not np.all(np.mod(order_host, 1) == 0):
        raise ValueError("the CUDA backend evaluates Bessel functions of integer order")

    if "jn" not in _kernels:
        _kernels["jn"] = cupy.ElementwiseKernel(
            "int32 n, float64 x",
            "float64 y",
            _BESSEL_J,
            "vecdiff_bessel_j",
            preamble="#include <cupy/xsf/cephes/j0.h>\n#include <cupy/xsf/cephes/j1.h>",
        )

    order = cupy.asarray(order).astype(cupy.int32)
    x = cupy.asarray(x, dtype=float)

    return _kernels["jn"](order, x)


def _convert(value, array):
    """Copy of a vecdiff object, list or tuple with every array passed through `array`.

    Only vecdiff's own objects are walked (fields, grids, frames, spectra), so
    matplotlib axes or user objects passed alongside are left untouched.
    Constructors are bypassed: they would place arrays on the current device.
    """
    if isinstance(value, (list, tuple)):
        return type(value)(_convert(item, array) for item in value)

    if isinstance(value, dict):
        return {key: _convert(item, array) for key, item in value.items()}

    if is_device_array(value) or (
        isinstance(value, np.ndarray) and value.dtype.kind in "biufc"
    ):
        return array(value)

    if type(value).__module__.startswith("vecdiff_") and hasattr(value, "__dict__"):
        state = vars(value)
        converted = {key: _convert(item, array) for key, item in state.items()}
        if all(converted[key] is state[key] for key in state):
            return value
        clone = object.__new__(type(value))
        for key, item in converted.items():
            object.__setattr__(clone, key, item)
        return clone

    return value


def host_copy(value):
    """The same vecdiff object with every CuPy array replaced by a NumPy copy."""
    return _convert(value, lambda a: a.get() if is_device_array(a) else a)


def device_copy(value):
    """The same vecdiff object with its arrays on the current device."""
    if _state["name"] == "CPU":
        return value

    return _convert(
        value, lambda a: a if is_device_array(a) else _state["module"].asarray(a)
    )


def _on_cpu_backend(function, args, kwargs):
    saved = dict(_state)
    _state.update(name="CPU", module=np)
    try:
        return function(*host_copy(args), **host_copy(kwargs))
    finally:
        _state.update(saved)


def on_host(function):
    """Run a plotting or output function on host copies under the CPU backend."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if _state["name"] == "CPU":
            return function(*args, **kwargs)
        return _on_cpu_backend(function, args, kwargs)

    return wrapper


def on_cpu(function):
    """Run a host-side solver on host copies under the CPU backend.

    Its results come back on the current device, so they combine with fields
    computed there.
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if _state["name"] == "CPU":
            return function(*args, **kwargs)
        return device_copy(_on_cpu_backend(function, args, kwargs))

    return wrapper


def cpu_reference(cls):
    """Class decorator: the constructor and public members run through ``on_cpu``."""
    for name, member in list(vars(cls).items()):
        if name.startswith("_") and name != "__init__":
            continue
        if isinstance(member, property):
            setattr(cls, name, property(on_cpu(member.fget)))
        elif isinstance(member, functools.cached_property):
            cached = functools.cached_property(on_cpu(member.func))
            cached.__set_name__(cls, name)
            setattr(cls, name, cached)
        elif isinstance(member, (classmethod, staticmethod)):
            setattr(cls, name, type(member)(on_cpu(member.__func__)))
        elif callable(member):
            setattr(cls, name, on_cpu(member))

    return cls


def set_backend(name):
    if name == "CPU":
        _state.update(name="CPU", module=np)
    elif name == "CUDA":
        _state.update(name="CUDA", module=_cupy())
    else:
        raise ValueError('backend must be "CPU" or "CUDA"')


def get_backend():
    return _state["name"]
