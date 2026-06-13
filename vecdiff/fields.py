import numpy as np
from .coordinate_transformation import (
    cartesian_to_circular, circular_to_cartesian,
    cartesian_to_polar, polar_to_cartesian
)


class Field:
    """
    Represents a sampled electromagnetic field over a 2D spatial grid.

    The canonical internal representation is always Cartesian (Ex, Ey).
    Other representations (circular, polar) are optionally available
    depending on the constructor used, or after calling
    generate_all_representations().

    The symmetry attribute records which representation is axially
    symmetric, enabling the correct Hankel-based propagation kernel.
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(self, Ex: np.ndarray, Ey: np.ndarray, grid,
                 symmetry: str | None = None, Ez: np.ndarray | None = None):
        self.x: np.ndarray = np.asarray(Ex, dtype=complex)
        self.y: np.ndarray = np.asarray(Ey, dtype=complex)
        self.z: np.ndarray | None = None if Ez is None else np.asarray(Ez, dtype=complex)
        self.grid = grid
        self.symmetry = symmetry

    @classmethod
    def from_cartesian(cls, Ex: np.ndarray, Ey: np.ndarray, grid,
                       symmetric: bool = False,
                       Ez: np.ndarray | None = None) -> "Field":
        return cls(Ex, Ey, grid, symmetry='cartesian' if symmetric else None, Ez=Ez)

    @classmethod
    def from_circular(cls, EL: np.ndarray, ER: np.ndarray, grid,
                      symmetric: bool = False,
                      Ez: np.ndarray | None = None) -> "Field":
        EL = np.asarray(EL, dtype=complex)
        ER = np.asarray(ER, dtype=complex)
        Ex, Ey = circular_to_cartesian(EL, ER)
        field = cls(Ex, Ey, grid, symmetry='circular' if symmetric else None, Ez=Ez)
        field.L, field.R = EL, ER
        return field

    @classmethod
    def from_polar(cls, Er: np.ndarray, Ephi: np.ndarray, grid,
                   symmetric: bool = False,
                   Ez: np.ndarray | None = None) -> "Field":
        Er = np.asarray(Er, dtype=complex)
        Ephi = np.asarray(Ephi, dtype=complex)
        Ex, Ey = polar_to_cartesian(Er, Ephi, grid.Phi)
        field = cls(Ex, Ey, grid, symmetry='polar' if symmetric else None, Ez=Ez)
        field.r, field.phi = Er, Ephi
        return field

    def generate_all_representations(self) -> None:
        """Compute and cache circular and polar components from (Ex, Ey)."""
        self.L, self.R   = cartesian_to_circular(self.x, self.y)
        self.r, self.phi = cartesian_to_polar(self.x, self.y, self.grid.Phi)

    def generate_Ez(self, wavelength, n=1.0, method="exact",
                    direction="+z", include_evanescent=False,
                    overwrite=True):
        """Generate and cache the longitudinal field component from transversality."""
        if self.z is not None and not overwrite:
            raise ValueError("Ez already exists. Use overwrite=True to regenerate it.")

        from .longitudinal import generate_Ez_field

        self.z = generate_Ez_field(
            self,
            wavelength=wavelength,
            n=n,
            method=method,
            direction=direction,
            include_evanescent=include_evanescent,
        )
        return self

    # ------------------------------------------------------------------ #
    #  Propagation                                                         #
    # ------------------------------------------------------------------ #

    def propagate_through_diopter(
        self,
        z: float,
        ovoid,
        q: np.ndarray | None = None,
        *,
        method: str = "auto",
        include_prefactor: bool = False,
        wavelength: float | None = None,
        output: str = "k",
    ) -> "Field":
        """Propagate the field through a refractive diopter to an observation plane."""
        if np.isclose(float(z), float(ovoid.zi)):
            if method not in {"auto", "hankel", "fft"}:
                raise ValueError("method must be one of: 'auto', 'hankel', 'fft'.")

            if method == "auto":
                method = "hankel" if self.symmetry is not None else "fft"

            if method == "hankel":
                if q is None:
                    raise ValueError("q is required for method='hankel'.")
                from .propagation import propagate_to_focal_plane_through_diopter
                return propagate_to_focal_plane_through_diopter(self, diopter=ovoid, q=q)

            from .propagation import propagate_to_focal_plane_through_diopter_fft
            return propagate_to_focal_plane_through_diopter_fft(
                self,
                diopter=ovoid,
                include_prefactor=include_prefactor,
                wavelength=wavelength,
                output=output,
            )
        else:
            raise NotImplementedError("Only propagation to the focal plane is supported.")


class FieldCartesian(Field):
    def __init__(self, x, y, grid=None, symmetric: bool = True, z=None):
        if grid is None:
            from .grid import Grid
            grid = Grid.from_polar(np.arange(np.asarray(x).size, dtype=float), np.array([0.0]))
        super().__init__(np.asarray(x, dtype=complex), np.asarray(y, dtype=complex),
                         grid, symmetry="cartesian" if symmetric else None, Ez=z)


class FieldCircular(Field):
    def __init__(self, L, R, grid=None, symmetric: bool = True, z=None):
        if grid is None:
            from .grid import Grid
            grid = Grid.from_polar(np.arange(np.asarray(L).size, dtype=float), np.array([0.0]))
        field = Field.from_circular(L, R, grid, symmetric=symmetric, Ez=z)
        self.__dict__.update(field.__dict__)


class FieldPolar(Field):
    def __init__(self, r, phi, grid=None, symmetric: bool = True, z=None):
        if grid is None:
            from .grid import Grid
            grid = Grid.from_polar(np.arange(np.asarray(r).size, dtype=float), np.array([0.0]))
        field = Field.from_polar(r, phi, grid, symmetric=symmetric, Ez=z)
        self.__dict__.update(field.__dict__)
