"""Lattice-sized arrays of the eikonal leg, held on the host and summed in row blocks.

A PlaneSpectrum keeps every mode's wavevector and three amplitudes, 96 bytes a
mode, and its checks make temporaries of the same size. A LatticeSpectrum
keeps the two transverse coefficients of each lattice mode, 32 bytes, in host
memory; wavevectors, the longitudinal component and the plane-wave sums are
formed one block of rows at a time on the active backend.
"""

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.fft

from vecdiff_general._backend import bd, get_backend, to_numpy
from vecdiff_general.fields import ElectricField, ElectromagneticField
from vecdiff_general.fourier.nufft import NUFFT
from vecdiff_general.media.medium import Z0

# Points per block of the streamed lattice computations: one block at a time
# on a GPU, several smaller ones at once on the host.
BLOCK = 1 << 20
HOST_BLOCK = 1 << 18
# Points per read of an incident field: a read costs most per call.
READ = 1 << 23
_ROUNDOFF = 64 * np.finfo(float).eps


def default_block():
    return BLOCK if get_backend() != "CPU" else HOST_BLOCK


def in_blocks(function, blocks):
    """function(block) for every block, in order.

    NumPy releases the interpreter lock in its array operations, so on the
    host the blocks run on several threads; on a GPU one after another.
    """
    workers = 1 if get_backend() != "CPU" else min(8, (os.cpu_count() or 2) // 2)
    if workers <= 1 or len(blocks) <= 1:
        yield from map(function, blocks)
        return
    with ThreadPoolExecutor(workers) as pool:
        running = deque()
        for block in blocks:
            running.append(pool.submit(function, block))
            if len(running) >= 2 * workers:
                yield running.popleft().result()
        while running:
            yield running.popleft().result()


def row_blocks(rows, width, size):
    """Slices of ``rows`` rows of ``width`` points, about ``size`` points each."""
    step = max(1, int(size) // max(int(width), 1))
    return [slice(start, min(start + step, rows)) for start in range(0, rows, step)]


def lattice_points(u, v, rows):
    """Points (u, v) of the lattice rows ``rows``, row-major as meshgrid(u, v)."""
    return bd.stack(bd.meshgrid(u, v[rows]), axis=-1).reshape(-1, 2)


def plane_blocks(expansion):
    """The modes of a materialised plane-wave expansion, as one block."""
    return [(expansion.wavevectors, expansion.amplitudes, expansion.origin)]


class PlaneReader:
    """Plane-wave sums at points of a plane, by a two-dimensional transform.

    On the plane origin + x u + y v every mode's phase k . (r - o) is
    k . (origin - o) + x k . u + y k . v, so its normal part folds into the
    amplitude: a type-3 sum in two dimensions instead of three. ``blocks`` is
    a list of (wavevectors, amplitudes, phase origin), prepared once, or a
    callable returning an iterable of them afresh for every call. FINUFFT
    runs on the host, so with method="nufft" the folded modes are kept there
    and only the sums return to the backend.
    """

    def __init__(self, blocks, origin, axes, *, components=3, method, **kwargs):
        self.origin, self.axes = bd.asarray(origin, float), bd.asarray(axes, float)
        self.components, self.method, self.options = components, method, kwargs
        # Prepared blocks replace the expansion they came from.
        self._blocks = blocks if callable(blocks) else None
        self._prepared = None if callable(blocks) else [self._fold(*b) for b in blocks]

    def _fold(self, wavevectors, amplitudes, origin):
        occupied = bd.any(amplitudes != 0, axis=-1)
        k = bd.asarray(wavevectors)[occupied]
        transverse = k @ self.axes
        if bd.any(transverse.imag != 0):
            raise ValueError("plane sums need real transverse wavevectors")
        phase = bd.exp(1j * (k @ (self.origin - bd.asarray(origin))))
        g, x = amplitudes[occupied] * phase[:, None], transverse.real
        if self.method == "nufft":
            return np.ascontiguousarray(to_numpy(g).T), to_numpy(x)
        return g, x

    def __call__(self, xy):
        """Sums at plane points xy (n, 2): shape (n, components)."""
        blocks = (
            self._prepared
            if self._prepared is not None
            else (self._fold(*b) for b in self._blocks())
        )
        if self.method != "nufft":
            xy = bd.asarray(xy, float)
            total = bd.zeros((len(xy), self.components), complex)
            for g, x in blocks:
                if len(x):
                    total += NUFFT(g, x, xy, sign=1, method=self.method, **self.options)
            return total

        import finufft

        from vecdiff_general.fourier.nufft import _threads

        xy = to_numpy(xy)
        targets = [np.ascontiguousarray(xy[:, axis]) for axis in (0, 1)]
        total = np.zeros((self.components, len(xy)), complex)
        for g, x in blocks:
            if len(x) and len(xy):
                total += finufft.nufft2d3(
                    np.ascontiguousarray(x[:, 0]),
                    np.ascontiguousarray(x[:, 1]),
                    g,
                    *targets,
                    isign=1,
                    eps=self.options.get("tolerance", 1e-12),
                    nthreads=_threads(self.options.get("threads")),
                )
        return bd.asarray(total.T)


def fft2_rows(values, block):
    """Uncentred 2D DFT of the components of host ``values`` (ny, nx, c), in place.

    Rows, then columns, are transformed one block at a time, so no temporary
    is larger than a block.
    """
    ny, nx = values.shape[:2]
    for rows in row_blocks(ny, nx, block):
        values[rows] = scipy.fft.fft(values[rows], axis=1, workers=-1)
    for columns in row_blocks(nx, ny, block):
        values[:, columns] = scipy.fft.fft(values[:, columns], axis=0, workers=-1)
    return values


class LatticeSpectrum:
    """Propagating plane waves of a lattice, held as two transverse coefficients each.

    E(r) = sum c[j, i] exp(i k[j, i] . (r - corner)) over the uncentred FFT
    lattice of ``grid`` (rows along v, columns along u), with ``corner`` its
    point [0, 0]. ``coefficients`` (ny, nx, 2) are the u and v components on
    the grid's axes, a host array; the normal component -kappa . c / k_z
    completes each mode in ``region``. Modes are formed and summed one
    block of rows at a time on the active backend.
    """

    def __init__(
        self,
        grid,
        region,
        wavelength0,
        coefficients,
        *,
        block=BLOCK,
        roundoff_removed_rms=0.0,
        evanescent_removed_rms=0.0,
    ):
        self.grid, self.region = grid, region
        self.wavelength0 = float(wavelength0)
        self.coefficients = coefficients
        self.block = int(block)
        self.roundoff_removed_rms = float(roundoff_removed_rms)
        self.evanescent_removed_rms = float(evanescent_removed_rms)
        if coefficients.shape != grid.shape + (2,):
            raise ValueError("coefficients must have grid.shape + (2,)")
        frame = grid.frame
        self.origin = frame.origin + grid.u[0] * frame.u + grid.v[0] * frame.v
        du, dv = (float(step) for step in grid.spacing)
        self._kx = 2 * np.pi * np.fft.fftfreq(len(grid.u), du)
        self._ky = 2 * np.pi * np.fft.fftfreq(len(grid.v), dv)

    @property
    def expansion(self):
        return self

    @property
    def wavenumber(self):
        return 2 * np.pi * self.region.material.n / self.wavelength0

    def relabeled(self, region):
        return LatticeSpectrum(
            self.grid,
            region,
            self.wavelength0,
            self.coefficients,
            block=self.block,
            roundoff_removed_rms=self.roundoff_removed_rms,
            evanescent_removed_rms=self.evanescent_removed_rms,
        )

    def blocks(self, magnetic=False):
        """(wavevectors, amplitudes, corner) of the occupied modes, a block of rows each.

        With ``magnetic`` the amplitudes hold E and then H, six columns.
        """
        ny, nx = self.grid.shape
        basis = self.grid.frame.basis
        k = self.wavenumber
        kx = bd.asarray(self._kx)
        for rows in row_blocks(ny, nx, self.block):
            c = bd.asarray(self.coefficients[rows]).reshape(-1, 2).astype(complex)
            ky = bd.repeat(bd.asarray(self._ky[rows]), nx)
            kxs = bd.tile(kx, rows.stop - rows.start)
            occupied = bd.any(c != 0, axis=-1)
            c, kxs, ky = c[occupied], kxs[occupied], ky[occupied]
            kz = bd.sqrt(k**2 - kxs**2 - ky**2)
            local = bd.stack(
                (c[:, 0], c[:, 1], -(kxs * c[:, 0] + ky * c[:, 1]) / kz), -1
            )
            wavevectors = bd.stack((kxs, ky, kz), -1) @ basis.T
            amplitudes = local @ basis.T
            if magnetic:
                k0 = 2 * np.pi / self.wavelength0
                field = bd.cross(wavevectors, amplitudes) / (k0 * Z0)
                amplitudes = bd.concatenate((amplitudes, field), axis=-1)
            yield wavevectors.astype(complex), amplitudes, self.origin

    def _sum(self, grid, magnetic, method, **kwargs):
        if method == "auto":
            import importlib.util

            method = "nufft" if importlib.util.find_spec("finufft") else "direct"
        points = bd.asarray(grid.points, float).reshape(-1, 3)
        basis = self.grid.frame.basis
        local = (points - self.origin) @ basis
        depth = local[:, 2]
        if len(points) and float(bd.max(depth) - bd.min(depth)) <= 1e-12 * (
            self.wavelength0
        ):
            # Every point on one plane parallel to the lattice: sum in two dimensions.
            plane = self.origin + float(depth[0]) * basis[:, 2]
            reader = PlaneReader(
                lambda: self.blocks(magnetic),
                plane,
                basis[:, :2],
                components=6 if magnetic else 3,
                method=method,
                **kwargs,
            )
            values = reader(local[:, :2])
        else:
            values = bd.zeros((len(points), 6 if magnetic else 3), complex)
            for wavevectors, amplitudes, origin in self.blocks(magnetic):
                if len(wavevectors):
                    values = values + NUFFT(
                        amplitudes,
                        wavevectors,
                        points - origin,
                        sign=1,
                        method=method,
                        **kwargs,
                    )
        return values.reshape(grid.shape + (values.shape[-1],))

    def electric(self, grid, *, method="auto", **kwargs):
        """E on any grid; keyword arguments select the Fourier summation method."""
        return ElectricField(
            grid, self.wavelength0, self._sum(grid, False, method, **kwargs)
        )

    def evaluate(self, grid, *, method="auto", **kwargs):
        """SI E and H on any grid."""
        values = self._sum(grid, True, method, **kwargs)
        electric = ElectricField(grid, self.wavelength0, values[..., :3])
        return ElectromagneticField(electric, values[..., 3:])


def lattice_spectrum(grid, region, wavelength0, values, *, cutoff, block=BLOCK):
    """LatticeSpectrum of carried u, v components ``values`` (host, grid.shape + (2,)).

    Transformed in place, like angular_spectrum with evanescent='discard':
    modes below roundoff of the largest are zeroed, then those on or beyond
    the light circle, and finally those beyond ``cutoff`` of it.
    """
    ny, nx = grid.shape
    values = fft2_rows(values.reshape(ny, nx, 2), block)
    values /= ny * nx
    k = 2 * np.pi * region.material.n / float(wavelength0)
    du, dv = (float(step) for step in grid.spacing)
    kx = 2 * np.pi * np.fft.fftfreq(nx, du)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dv)
    blocks = row_blocks(ny, nx, block)
    largest = max(
        float(np.max(np.sqrt(np.sum(abs(values[rows]) ** 2, -1)))) for rows in blocks
    )
    roundoff = evanescent = 0.0
    for rows in blocks:
        part = values[rows]
        size = np.sqrt(np.sum(abs(part) ** 2, -1))
        small = size < _ROUNDOFF * largest
        roundoff += float(np.sum(size[small] ** 2))
        part[small] = 0
        kappa2 = ky[rows, None] ** 2 + kx[None] ** 2
        grazing = abs(k**2 - kappa2) <= _ROUNDOFF * k**2
        beyond = (kappa2 > k**2) | grazing
        evanescent += float(np.sum(abs(part[beyond]) ** 2))
        part[beyond | (kappa2 > (cutoff * k) ** 2)] = 0

    return LatticeSpectrum(
        grid,
        region,
        wavelength0,
        values,
        block=block,
        roundoff_removed_rms=np.sqrt(roundoff),
        evanescent_removed_rms=np.sqrt(evanescent),
    )
