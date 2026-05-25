# CartesianSurfaces.py

import numpy as np

from .GOTS_parameters import GOTS_params


class CartesianSurface:
    def __init__(self, n0, ni, z0, zi):
        self.n0 = n0
        self.ni = ni
        self.z0 = z0
        self.zi = zi

        
        (
        self.G, 
        self.O, 
        self.T, 
        self.S 
        ) = GOTS_params(self.n0, self.z0, self.ni, self.zi)

    def z(self, rho):
        rho = np.asarray(rho, dtype=float)

        G = self.G
        O = self.O
        T = self.T
        S = self.S

        return ((O + T*rho**2)*rho**2) / (
            1
            + S*rho**2
            + np.sqrt(1 + (2*S - O**2*G)*rho**2)
        )

    def dz(self, rho):
        rho = np.asarray(rho, dtype=float)

        G = self.G
        O = self.O
        T = self.T
        S = self.S

        Q = np.sqrt(1 + (2*S - O**2*G)*rho**2)

        N = (O + T*rho**2)*rho**2
        D = 1 + S*rho**2 + Q

        Np = 2*O*rho + 4*T*rho**3
        Qp = (2*S - O**2*G)*rho / Q
        Dp = 2*S*rho + Qp

        return (Np*D - N*Dp) / D**2

    def r(self, rho):
        rho = np.asarray(rho, dtype=float)
        z = self.z(rho)
        return np.sqrt(rho**2 - z**2)
