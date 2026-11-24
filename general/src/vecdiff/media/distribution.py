"""Assignment of constitutive matter to geometrical volumes."""

from dataclasses import dataclass

from vecdiff._backend import bd
from vecdiff._printing import block, guarded, shorten
from vecdiff.media.medium import Medium


@dataclass(frozen=True)
class MaterialRegion:
    name: str
    geometry: object
    material: Medium

    def __post_init__(self):
        if not callable(getattr(self.geometry, "contains", None)):
            raise TypeError("material region geometry must provide spatial membership")

        if not isinstance(self.material, Medium):
            raise TypeError("a material region needs a media.Medium")

    @guarded
    def __repr__(self):
        """Its name, its volume and its material."""
        return shorten(
            f"MaterialRegion({self.name!r}, {self.geometry!r}, {self.material!r})"
        )


@dataclass(frozen=True)
class MaterialDistribution:
    background: Medium
    regions: tuple[MaterialRegion, ...] = ()

    def __post_init__(self):
        if not isinstance(self.background, Medium):
            raise TypeError("background must be a media.Medium")

        object.__setattr__(self, "regions", tuple(self.regions))

    @guarded
    def __repr__(self):
        """Its background and the names of its regions."""
        names = ", ".join(repr(region.name) for region in self.regions)
        return shorten(f"MaterialDistribution({self.background!r}, regions=[{names}])")

    @guarded
    def __str__(self):
        """The background and each region, its volume and material."""
        rows = [("background", repr(self.background))]
        rows += [
            (region.name, f"{region.material!r} in {region.geometry!r}")
            for region in self.regions
        ]
        return block("MaterialDistribution", rows)

    def refractive_index(self, points):
        """Piecewise n(r); overlapping bodies require an explicit topology."""
        points = bd.asarray(points, dtype=float)

        if points.shape[-1:] != (3,) or not bd.isfinite(points).all():
            raise ValueError("positions must be finite Cartesian coordinates")

        materials = (self.background, *(r.material for r in self.regions))
        complex_index = any(m.absorbing for m in materials)
        indices = bd.full(
            points.shape[:-1],
            self.background.n,
            dtype=complex if complex_index else float,
        )
        occupied = bd.zeros(points.shape[:-1], dtype=bool)

        for region in self.regions:
            inside = region.geometry.contains(points)

            if bd.any(occupied & inside):
                raise ValueError(
                    "overlapping material regions have no unique assignment"
                )

            indices[inside] = region.material.n
            occupied |= inside

        return indices
