import numpy as np


class Grid:
    """Sampled coordinates, optionally tagged with the surface they live on.

    ``reference`` is the second half of the pair ``(r, E(r))``: it records
    whether the samples sit on a reference sphere or on a tangent plane, so the
    propagators can stop guessing.  ``None`` means the samples are labelled by
    the surface radius directly, which for a pupil field is the same as living
    on the incident sphere ``G``.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        domain: str = "space",
        dual: "Grid | None" = None,
        reference=None,
    ):
        self.X = np.asarray(X, dtype=float)
        self.Y = np.asarray(Y, dtype=float)
        self.type = "cartesian"
        self.domain = domain
        self.dual = dual
        self.reference = reference
        self.generate_polar_from_cartesian()

    @classmethod
    def from_cartesian(
        cls,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        domain: str = "space",
        dual: "Grid | None" = None,
        reference=None,
    ) -> "Grid":
        return cls(X, Y, domain=domain, dual=dual, reference=reference)

    @classmethod
    def from_axes(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        *,
        domain: str = "space",
        dual: "Grid | None" = None,
        reference=None,
    ) -> "Grid":
        """Return a Cartesian grid from one-dimensional x and y axes."""
        X, Y = np.meshgrid(np.asarray(x, dtype=float), np.asarray(y, dtype=float), indexing="xy")
        return cls(X, Y, domain=domain, dual=dual, reference=reference)

    @classmethod
    def from_spacing(
        cls,
        shape: tuple[int, int],
        dx: float = 1.0,
        dy: float = 1.0,
        *,
        domain: str = "space",
        dual: "Grid | None" = None,
        reference=None,
    ) -> "Grid":
        """Return a centered Cartesian grid with sample spacings ``dx`` and ``dy``."""
        ny, nx = shape
        x = (np.arange(nx) - nx//2)*dx
        y = (np.arange(ny) - ny//2)*dy
        X, Y = np.meshgrid(x, y, indexing="xy")
        return cls(X, Y, domain=domain, dual=dual, reference=reference)

    @classmethod
    def from_polar(cls, r: np.ndarray, varphi: np.ndarray | None = None, *, reference=None) -> "Grid":
        r = np.asarray(r, dtype=float)
        grid = cls.__new__(cls)
        if varphi is None:
            grid.R = r
            grid.Phi = np.zeros_like(r)
            grid.r = r
            grid.varphi = np.array([0.0])
        else:
            varphi = np.asarray(varphi, dtype=float)
            grid.r = r
            grid.varphi = varphi
            grid.R, grid.Phi = np.meshgrid(r, varphi)
        grid.type = "polar"
        grid.domain = "space"
        grid.dual = None
        grid.reference = reference
        grid.generate_cartesian_from_polar()
        return grid

    @property
    def shape(self) -> tuple[int, int]:
        return self.X.shape

    @property
    def dx(self) -> float:
        return self.spacing()[0]

    @property
    def dy(self) -> float:
        return self.spacing()[1]

    def spacing(self) -> tuple[float, float]:
        """Return uniform Cartesian spacings ``(dx, dy)``."""
        self._require_cartesian_grid()
        return (
            self._uniform_spacing(self.X, axis=1, name="x"),
            self._uniform_spacing(self.Y, axis=0, name="y"),
        )

    def kgrid(self, shape: int | tuple[int, int] | None = None, *, shift: bool = True) -> "Grid":
        """Return the angular-frequency grid dual to this Cartesian grid."""
        self._require_cartesian_grid()
        if shape is None:
            ny, nx = self.shape
        elif hasattr(shape, "__index__"):
            ny = nx = shape.__index__()
        else:
            ny, nx = shape

        dx, dy = self.spacing()

        kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
        if shift:
            kx = np.fft.fftshift(kx)
            ky = np.fft.fftshift(ky)

        KX, KY = np.meshgrid(kx, ky, indexing="xy")
        return Grid.from_cartesian(KX, KY, domain="k", dual=self, reference=self.reference)

    def generate_cartesian_from_polar(self) -> None:
        self.X = self.R * np.cos(self.Phi)
        self.Y = self.R * np.sin(self.Phi)

    def generate_polar_from_cartesian(self) -> None:
        self.R   = np.sqrt(self.X**2 + self.Y**2)
        self.Phi = np.arctan2(self.Y, self.X)

    def _require_cartesian_grid(self) -> None:
        if self.type != "cartesian":
            raise ValueError("Fourier grids require a Cartesian Grid.")
        if self.X.shape != self.Y.shape:
            raise ValueError("X and Y grids must have the same shape.")
        if self.X.ndim != 2:
            raise ValueError("X and Y must be 2D arrays.")

    @staticmethod
    def _uniform_spacing(values: np.ndarray, axis: int, name: str) -> float:
        diffs = np.diff(values, axis=axis)
        if diffs.size == 0:
            raise ValueError(f"{name} grid axis must contain at least two samples.")

        spacing = float(np.mean(diffs))
        if not np.allclose(diffs, spacing):
            raise ValueError(f"{name} grid axis must be uniformly sampled.")
        if np.isclose(spacing, 0.0):
            raise ValueError(f"{name} grid spacing must be nonzero.")
        return spacing
