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


    def ts(self, r):
        o = self.ovoid

        r = np.asarray(r, dtype=float)

        z  = o.z(r)
        zp = o.dz(r)
        r  = o.r(r)

        l0 = np.sqrt((o.z0 - z)**2 + r**2)
        li = np.sqrt((o.zi - z)**2 + r**2)

        A0 = zp*(r**2 - z*o.z0) - r*(z - o.z0)
        Ai = zp*(r**2 - z*o.zi) - r*(z - o.zi)

        eta = o.ni / o.n0

        return 2*A0 / (A0 + eta*(l0/li)*Ai)

    def tp(self, r):
        o = self.ovoid

        r = np.asarray(r, dtype=float)

        z  = o.z(r)
        zp = o.dz(r)
        r  = o.r(r)

        l0 = np.sqrt((o.z0 - z)**2 + r**2)
        li = np.sqrt((o.zi - z)**2 + r**2)

        A0 = zp*(r**2 - z*o.z0) - r*(z - o.z0)
        Ai = zp*(r**2 - z*o.zi) - r*(z - o.zi)

        eta = o.ni / o.n0

        return 2*A0 / (eta*A0 + (l0/li)*Ai)




# TODO : Implementar El Calculo de las Bases.