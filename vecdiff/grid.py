import numpy as np

class Grid:
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = np.asarray(X, dtype=float)
        self.Y = np.asarray(Y, dtype=float)
        self.type = "cartesian"
        self.generate_polar_from_cartesian()

    @classmethod
    def from_cartesian(cls, X: np.ndarray, Y: np.ndarray) -> "Grid":
        return cls(X, Y)

    @classmethod
    def from_polar(cls, r: np.ndarray, varphi: np.ndarray | None = None) -> "Grid":
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
        grid.generate_cartesian_from_polar()
        return grid

    def generate_cartesian_from_polar(self) -> None:
        self.X = self.R * np.cos(self.Phi)
        self.Y = self.R * np.sin(self.Phi)

    def generate_polar_from_cartesian(self) -> None:
        self.R   = np.sqrt(self.X**2 + self.Y**2)
        self.Phi = np.arctan2(self.Y, self.X)
