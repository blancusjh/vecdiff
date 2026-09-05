"""An ordered assembly of interfaces bounding homogeneous regions."""
from dataclasses import dataclass
from .dielectric_interface import DielectricInterface


@dataclass(frozen=True)
class InterfaceAssembly:
    interfaces: tuple

    def __post_init__(self):
        interfaces = tuple(self.interfaces)
        if not interfaces or not all(isinstance(i, DielectricInterface) for i in interfaces):
            raise ValueError("an assembly requires dielectric interfaces")
        for left, right in zip(interfaces, interfaces[1:]):
            if left.transmitted_medium != right.incident_medium:
                raise ValueError("adjacent interfaces must bound the same medium")
        object.__setattr__(self, "interfaces", interfaces)

    @property
    def media(self):
        return (self.interfaces[0].incident_medium,) + tuple(i.transmitted_medium for i in self.interfaces)
