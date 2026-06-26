from scipy.special import jv
import matplotlib.pyplot as plt
import numpy as np

from vecdiff.view import plot_radial_field



modes = [0, 1, 2]

# PARAMETERS 

L = 20.0
N = 500



for m in modes: 

    plot_radial_field(
    lambda r: np.where(r == 0, 0.0, (jv(m, r))**2 / r**(m * 2)),
    L=L,
    N=N,
    title=r"$j_\\nu(r)$"
)



rr = np.linspace(-L, L, N) 


for m in modes :

    plt.plot(rr, jv(m, rr))
    plt.show() 



