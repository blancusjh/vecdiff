import numpy as np

from .coordinate_transformation import cartesian_to_polar, circular_to_polar


def radial_component(field, varphi):
    p = field.polarization
    if p == "polar":
        return field.r
    if p == "cartesian":
        er, _ = cartesian_to_polar(field.x, field.y, varphi)
        return er
    if p == "circular":
        er, _ = circular_to_polar(field.L, field.R, varphi)
        return er
    raise ValueError("Unknown polarization mode.")


def reconstruct_longitudinal_component(field, radius, z_ref, sag_function, varphi, eps=1e-12):
    rr = np.asarray(radius, dtype=float)
    er = radial_component(field, varphi)
    sag = np.asarray(sag_function(rr), dtype=float)
    denom = z_ref - sag
    safe = np.where(np.abs(denom) < eps, np.where(denom >= 0.0, eps, -eps), denom)
    return (rr / safe) * er


class Field:
    """
    Class that represent a general field determined by two components. 
    The Field class has a method to propagate the field at a distance z from the original plane. 
    """
    polarization = "generic"

    def __init__(self, component1, component2):
        self.component1 = np.asarray(component1, 
                                     dtype=complex)
        self.component2 = np.asarray(component2, 
                                     dtype=complex)

        self.components = np.array([self.component1, self.component2])

    def propagate(self, z, diopter, observation):
        zi = float(diopter.zi)

        if np.isclose(float(z), zi):
            from .propagation import propagate_to_focal_plane_through_diopter
            return propagate_to_focal_plane_through_diopter(self, diopter=diopter, observation=observation)
        
        raise NotImplementedError("Only z == diopter.zi is implemented. " \
        "Use focal propagation for now.")



# Fields in its different Representations. 
class FieldCircular(Field):
    polarization = "circular"

    def __init__(self, L, R):
        super().__init__(L, R)
        self.L = self.component1
        self.R = self.component2


class FieldCartesian(Field):
    polarization = "cartesian"

    def __init__(self, x, y):
        super().__init__(x, y)
        self.x = self.component1
        self.y = self.component2


class FieldPolar(Field):
    polarization = "polar"

    def __init__(self, r, phi):
        super().__init__(r, phi)
        self.r = self.component1
        self.phi = self.component2
