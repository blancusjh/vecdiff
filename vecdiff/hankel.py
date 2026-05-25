import numpy as np

from scipy.special import jv, j1
from scipy.integrate import simpson


# __DEFAULT VALUES__

N = 500 


def integrate(f, x, method="simpson"): 
    if method == 'simpson': 
        return simpson(f, x=x)



class HankelTransform:

    @staticmethod
    def radial_grid(R, N=N):
        return np.linspace(0.0, R, N)

    @staticmethod
    def transform(order, r, f_r, q):
        # Performs Hankel Transform of m order 
        kernel = jv(order, q*r)*r

        return integrate(f_r*kernel, x=r)

    @staticmethod
    def transform_array(order, rho, f_rho, q_array):

        out = np.empty_like(q_array, dtype=complex)

        for i, q in enumerate(q_array):

            out[i] = HankelTransform.transform(
                order,
                rho,
                f_rho,
                q
            )

        return out


def HT_N(m, f, R, N):

    # m : float, orden. 
    # func : Function of r. 
    # R : Maximum Radius. 
    # N: Number of samples.

    # inicializa grilla radial
    rr   = HankelTransform.radial_grid(R, N)
    
    HT = HankelTransform.transform_array

    q = np.linspace(1e-3, 10.0, 500)
    F = HT(m, rr, f(rr), q)

    return F, q
