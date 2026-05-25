import numpy as np



# Polar - Cartesian 

def cartesian_to_polar(Ex, Ey, varphi):
    Er = Ex * np.cos(varphi) + Ey * np.sin(varphi)
    Ephi = -Ex * np.sin(varphi) + Ey * np.cos(varphi)
    return Er, Ephi


def polar_to_cartesian(Er, Ephi, varphi):
    Ex = Er * np.cos(varphi) - Ephi * np.sin(varphi)
    Ey = Er * np.sin(varphi) + Ephi * np.cos(varphi)
    return Ex, Ey


# Circular - Cartesian 

def circular_to_cartesian(EL, ER):
    Ex = (EL + ER) / np.sqrt(2)
    Ey = -1j * (EL - ER) / np.sqrt(2)
    return Ex, Ey


def cartesian_to_circular(Ex, Ey):
    EL = (Ex + 1j * Ey) / np.sqrt(2)
    ER = (Ex - 1j * Ey) / np.sqrt(2)
    return EL, ER


# Circular - Polar 

def circular_to_polar(EL, ER, varphi):
    Er = (exp(-1j * varphi) * EL + exp(1j * varphi) * ER) / np.sqrt(2)
    Ephi = (-1j * exp(-1j * varphi) * EL + 1j * exp(1j * varphi) * ER) / np.sqrt(2)
    return Er, Ephi


def polar_to_circular(Er, Ephi, varphi):
    EL = exp(1j * varphi) * (Er + 1j * Ephi) / np.sqrt(2)
    ER = exp(-1j * varphi) * (Er - 1j * Ephi) / np.sqrt(2)
    return EL, ER

