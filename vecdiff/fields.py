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
                 symmetry: str | None = None):
        self.x: np.ndarray = np.asarray(Ex, dtype=complex)
        self.y: np.ndarray = np.asarray(Ey, dtype=complex)
        self.grid = grid
        self.symmetry = symmetry

    @classmethod
    def from_cartesian(cls, Ex: np.ndarray, Ey: np.ndarray, grid,
                       symmetric: bool = False) -> "Field":
        return cls(Ex, Ey, grid, symmetry='cartesian' if symmetric else None)

    @classmethod
    def from_circular(cls, EL: np.ndarray, ER: np.ndarray, grid,
                      symmetric: bool = False) -> "Field":
        EL = np.asarray(EL, dtype=complex)
        ER = np.asarray(ER, dtype=complex)
        Ex, Ey = circular_to_cartesian(EL, ER)
        field = cls(Ex, Ey, grid, symmetry='circular' if symmetric else None)
        field.L, field.R = EL, ER
        return field

    @classmethod
    def from_polar(cls, Er: np.ndarray, Ephi: np.ndarray, grid,
                   symmetric: bool = False) -> "Field":
        Er = np.asarray(Er, dtype=complex)
        Ephi = np.asarray(Ephi, dtype=complex)
        Ex, Ey = polar_to_cartesian(Er, Ephi, grid.Phi)
        field = cls(Ex, Ey, grid, symmetry='polar' if symmetric else None)
        field.r, field.phi = Er, Ephi
        return field

    def generate_all_representations(self) -> None:
        """Compute and cache circular and polar components from (Ex, Ey)."""
        self.L, self.R   = cartesian_to_circular(self.x, self.y)
        self.r, self.phi = cartesian_to_polar(self.x, self.y, self.grid.Phi)

    # ------------------------------------------------------------------ #
    #  Propagation                                                         #
    # ------------------------------------------------------------------ #

    def propagate_through_diopter(self, z: float, ovoid, q: np.ndarray) -> "Field":
        """Propagate the field through a refractive diopter to an observation plane."""
        if np.isclose(float(z), float(ovoid.zi)):
            from .propagation import propagate_to_focal_plane_through_diopter
            return propagate_to_focal_plane_through_diopter(self, diopter=ovoid, q=q)
        else:
            raise NotImplementedError("Only propagation to the focal plane is supported.")


class FieldCartesian(Field):
    def __init__(self, x, y, grid=None, symmetric: bool = True):
        if grid is None:
            from .grid import Grid
            grid = Grid.from_polar(np.arange(np.asarray(x).size, dtype=float), np.array([0.0]))
        super().__init__(np.asarray(x, dtype=complex), np.asarray(y, dtype=complex),
                         grid, symmetry="cartesian" if symmetric else None)


class FieldCircular(Field):
    def __init__(self, L, R, grid=None, symmetric: bool = True):
        if grid is None:
            from .grid import Grid
            grid = Grid.from_polar(np.arange(np.asarray(L).size, dtype=float), np.array([0.0]))
        field = Field.from_circular(L, R, grid, symmetric=symmetric)
        self.__dict__.update(field.__dict__)


class FieldPolar(Field):
    def __init__(self, r, phi, grid=None, symmetric: bool = True):
        if grid is None:
            from .grid import Grid
            grid = Grid.from_polar(np.arange(np.asarray(r).size, dtype=float), np.array([0.0]))
        field = Field.from_polar(r, phi, grid, symmetric=symmetric)
        self.__dict__.update(field.__dict__)
