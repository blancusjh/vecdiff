"""Batched, normalized multivariate Taylor algebra for local optical transport.

Coefficients multiply u**i v**j s**k (derivatives divided by factorials).
Only total degrees up to the declared depth are stored. No FFT or finite
difference is used to obtain derivatives of the interface coefficients.
"""

from functools import lru_cache

import numpy as np

try:
    from ._taylor_core import taylor_product
except ImportError:
    taylor_product = None


@lru_cache(None)
def _indices(degree):
    indices = tuple(
        (i, j, d - i - j)
        for d in range(degree + 1)
        for i in range(d + 1)
        for j in range(d - i + 1)
    )
    lookup = {key: i for i, key in enumerate(indices)}
    left, right, starts = [], [], []
    for alpha in indices:
        starts.append(len(left))
        for i in range(alpha[0] + 1):
            for j in range(alpha[1] + 1):
                for k in range(alpha[2] + 1):
                    beta = (i, j, k)
                    left.append(lookup[beta])
                    right.append(lookup[tuple(a - b for a, b in zip(alpha, beta))])
    return indices, lookup, np.array(left), np.array(right), np.array(starts)


class TaylorSpace:
    def __init__(self, degree, count):
        self.degree, self.count = degree, count
        self.indices, self.lookup, self.left, self.right, self.starts = _indices(degree)
        self.powers = np.array(self.indices)
        self.native_indices = tuple(
            np.asarray(v + 1, np.int32)
            for v in (self.left, self.right, np.r_[self.starts, len(self.left)])
        )
        self.derivatives = []
        for axis in range(3):
            source = np.flatnonzero(self.powers[:, axis] > 0)
            target = [
                self.lookup[tuple(p - np.eye(3, dtype=int)[axis])]
                for p in self.powers[source]
            ]
            self.derivatives.append(
                (source, np.array(target), self.powers[source, axis])
            )

    def constant(self, value):
        a = np.zeros((len(self.indices), self.count), complex)
        a[0] = value
        return Taylor(self, a)

    def variable(self, axis, center=0):
        result = self.constant(center)
        key = tuple(np.eye(3, dtype=int)[axis])
        result.c[self.lookup[key]] = 1
        return result


class Taylor:
    __array_priority__ = 10000

    def __init__(self, space, coefficients):
        self.space, self.c = space, coefficients

    def _coerce(self, other):
        if isinstance(other, Taylor):
            if other.space is not self.space:
                raise ValueError("Taylor operands must share a space")
            return other
        return self.space.constant(other)

    def __add__(self, other):
        return Taylor(self.space, self.c + self._coerce(other).c)

    __radd__ = __add__

    def __neg__(self):
        return Taylor(self.space, -self.c)

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        if not isinstance(other, Taylor):
            return Taylor(self.space, self.c * other)
        other = self._coerce(other)
        s = self.space
        if taylor_product is None:
            result = np.add.reduceat(
                self.c[s.left] * other.c[s.right], s.starts, axis=0
            )
        else:
            result = taylor_product(self.c.T, other.c.T, *s.native_indices).T
        return Taylor(s, result)

    __rmul__ = __mul__

    def reciprocal(self):
        if np.any(abs(self.c[0]) < 1e-14):
            raise ValueError("singular Taylor denominator; grazing/critical regime")
        # Newton doubling: correct through degree 2p+1 if correct through p.
        out = self.space.constant(1 / self.c[0])
        for _ in range(int(np.ceil(np.log2(self.space.degree + 1)))):
            out = out * (2 - self * out)
        return out

    def __truediv__(self, other):
        if not isinstance(other, Taylor):
            return Taylor(self.space, self.c / other)
        return self * other.reciprocal()

    def __rtruediv__(self, other):
        return self.reciprocal() * other

    def __pow__(self, power):
        if power == 0.5:
            if np.any(self.c[0].real <= 1e-14) or np.any(abs(self.c[0].imag) > 1e-13):
                raise ValueError("real noncritical propagating Taylor branch required")
            # Binomial expansion about a nonzero constant; exact truncated series.
            root = np.sqrt(self.c[0])
            delta = self / self.c[0] - 1
            term = self.space.constant(1)
            out = term
            coefficient = 1.0
            for j in range(1, self.space.degree + 1):
                term = term * delta
                coefficient *= (1.5 - j) / j
                out = out + coefficient * term
            return out * root
        if not isinstance(power, int) or power < 0:
            raise ValueError("supported powers are nonnegative integers and 0.5")
        out = self.space.constant(1)
        for _ in range(power):
            out = out * self
        return out

    def derivative(self, axis):
        source, target, factors = self.space.derivatives[axis]
        out = np.zeros_like(self.c)
        out[target] = factors[:, None] * self.c[source]
        return Taylor(self.space, out)

    def cos(self):
        center = self.c[0]
        delta = self - center
        term = self.space.constant(1)
        values = (np.cos(center), -np.sin(center), -np.cos(center), np.sin(center))
        out = self.space.constant(values[0])
        for j in range(1, self.space.degree + 1):
            term = term * delta / j
            out = out + term * values[j % 4]
        return out

    def trace(self):
        out = self.c.copy()
        out[self.space.powers[:, 2] > 0] = 0
        return Taylor(self.space, out)

    def truncate(self, degree):
        out = self.c.copy()
        out[self.space.powers.sum(axis=1) > degree] = 0
        return Taylor(self.space, out)

    def set_normal_coefficient(self, rhs, power, degree):
        s = self.space
        source = np.flatnonzero(
            (s.powers[:, 2] == power - 1) & (s.powers.sum(axis=1) < degree)
        )
        target = [s.lookup[tuple(p + [0, 0, 1])] for p in s.powers[source]]
        self.c[target] = rhs.c[source] / power


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def solve_trace(matrix, rhs, degree):
    """Taylor coefficient solve using only the constant 6x6 factorization."""
    space = rhs[0].space
    m = np.stack(
        [np.stack(row, axis=-1) for row in [[v.c for v in row] for row in matrix]],
        axis=-2,
    )
    b = np.stack([v.c for v in rhs], axis=-1)
    inverse = np.linalg.inv(m[0])
    answer = np.zeros_like(b)
    for i, alpha in enumerate(space.indices):
        if alpha[2] or sum(alpha) > degree:
            continue
        start = space.starts[i]
        stop = space.starts[i + 1] if i + 1 < len(space.starts) else len(space.left)
        left, right = space.left[start:stop], space.right[start:stop]
        keep = left != 0
        remainder = np.einsum("tbij,tbj->bi", m[left[keep]], answer[right[keep]])
        answer[i] = np.einsum("bij,bj->bi", inverse, b[i] - remainder)
    return [Taylor(space, answer[..., i]) for i in range(6)]
