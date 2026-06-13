import numpy as np




# __ DEFAULT PARAMETERS ___

n0 = 1.0
ni = 1.5
z0 = 5
zi = 10


from .CartesianSurfaces import CartesianSurface


# === PARAXIAL FRESNEL COEFFICIENTS ===

class FresnelOvoidParax:
    '''
    Provides Fresnel Coeffients for Ovoids defined by its physical parameters in 
    the paraxial aproximation.
    '''
    def __init__(self, n0, ni, z0, zi):
        self.n0 = n0
        self.ni = ni
        self.z0 = z0
        self.zi = zi

        self.gamma_0 = 2 * n0 / (n0 + ni)
        self.xi = (
            n0 * (z0 - zi) ** 2
            / (z0**2 * zi**2 * (n0**2 - ni**2))
        )

    def t_s(self, r):
        return self.gamma_0 - self.ni * self.xi * r**2

    def t_p(self, r):
        return self.gamma_0 + self.n0 * self.xi * r**2

    def coeffs(self, r):
        return self.t_s(r), self.t_p(r)



# Exact Fresnel Coefficeints for the Ovoid
class FresnelOvoid:

    def __init__(self, ovoid=None, n0=None, z0=None, ni=None, zi=None, no=None, zo=None):

        if ovoid is not None: 
            self.ovoid = ovoid
        else: 
            # Backward compatibility for legacy parameter names no/zo.
            if n0 is None and no is not None:
                n0 = no
            if z0 is None and zo is not None:
                z0 = zo
            self.ovoid = CartesianSurface(n0=n0, z0=z0, ni=ni, zi=zi)

    def _cosines(self, rho):
        """Return positive incidence/transmission cosines on the Cartesian surface.

        The Cartesian surface is parametrized by rho.  Its cylindrical radius is
        r(rho)=sqrt(rho**2-z(rho)**2).  Keeping rho and r(rho) distinct avoids
        the sign/variable ambiguity that made the FFT and Hankel branches use
        different effective Fresnel factors, especially for the physical z0 < 0
        convention.
        """
        o = self.ovoid
        rho = np.asarray(rho, dtype=float)

        z = o.z(rho)
        dz = o.dz(rho)
        r = o.r(rho)

        dr = np.divide(
            rho - z * dz,
            r,
            out=np.ones_like(rho, dtype=float),
            where=np.abs(r) > 1e-13,
        )

        normal_norm = np.hypot(dr, dz)
        N_r = -dz / normal_norm
        N_z = dr / normal_norm

        l0 = np.hypot(r, z - o.z0)
        li = np.hypot(r, o.zi - z)

        u0_r = r / l0
        u0_z = (z - o.z0) / l0
        ui_r = -r / li
        ui_z = (o.zi - z) / li

        cos_i = np.abs(u0_r * N_r + u0_z * N_z)
        cos_t = np.abs(ui_r * N_r + ui_z * N_z)
        return np.clip(cos_i, 0.0, 1.0), np.clip(cos_t, 0.0, 1.0)


    def ts(self, r):
        o = self.ovoid
        cos_i, cos_t = self._cosines(r)
        return 2.0 * o.n0 * cos_i / (o.n0 * cos_i + o.ni * cos_t)

    def tp(self, r):
        o = self.ovoid
        cos_i, cos_t = self._cosines(r)
        return 2.0 * o.n0 * cos_i / (o.ni * cos_i + o.n0 * cos_t)




# TODO : Implementar El Calculo de las Bases.
