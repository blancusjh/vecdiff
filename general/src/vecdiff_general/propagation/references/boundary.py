"""Closed boundaries and quadrature for dielectric reference problems."""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

from ...sampling import SurfaceGrid
from ...volumes import SphereVolume, VolumeBetweenSurfaces


@dataclass(frozen=True)
class Meridian:
    """Axisymmetric boundary nodes (rho,z), outward normals and area / 2pi."""

    points: np.ndarray
    normals: np.ndarray
    weights: np.ndarray

    def surface(self, azimuths=16, *, phase=0.0):
        phi = phase + np.arange(azimuths) * 2 * np.pi / azimuths
        cp, sp = np.cos(phi), np.sin(phi)
        r, z = self.points.T
        nr, nz = self.normals.T
        points = np.stack(
            np.broadcast_arrays(r[:, None] * cp, r[:, None] * sp, z[:, None]), -1
        )
        normals = np.stack(
            np.broadcast_arrays(nr[:, None] * cp, nr[:, None] * sp, nz[:, None]), -1
        )
        weights = np.broadcast_to(
            self.weights[:, None] * 2 * np.pi / azimuths, points.shape[:-1]
        )
        return SurfaceGrid(points, normals, weights)


def boundary_meridian(volume, count):
    """Gauss quadrature including both lens faces and the cylindrical rim.

    Axis is global z; translated z vertices are supported. count nodes per
    face/rim for a lens, count nodes over the semicircle for a sphere.
    """
    if not isinstance(count, (int, np.integer)) or count < 2:
        raise ValueError("boundary count must be an integer >= 2")
    x, w = leggauss(count)
    t, weights = (x + 1) / 2, w / 2
    if isinstance(volume, SphereVolume):
        if np.any(volume.center[:2] != 0):
            raise ValueError("axisymmetric boundary must be centred on global z")
        theta = np.pi * t
        normals = np.stack((np.sin(theta), np.cos(theta)), -1)
        points = volume.radius * normals + [0, volume.center[2]]
        return Meridian(
            points, normals, weights * np.pi * volume.radius**2 * np.sin(theta)
        )
    if not isinstance(volume, VolumeBetweenSurfaces):
        raise TypeError("expected SphereVolume or VolumeBetweenSurfaces")
    points, normals, areas = [], [], []
    r = volume.rim_radius * t
    xy = np.stack((r, np.zeros_like(r)), -1)
    for surface, sign in ((volume.front, -1), (volume.back, 1)):
        if not np.allclose(surface.frame.basis, np.eye(3)) or np.any(
            surface.frame.origin[:2] != 0
        ):
            raise ValueError("reference lens must be coaxial with global z")
        # Validate rotational invariance, including the azimuthal derivative.
        rotated = xy[:, ::-1]
        if not np.allclose(
            surface.height(xy), surface.height(rotated), rtol=1e-10, atol=1e-14
        ) or not np.allclose(
            surface.gradient(xy),
            surface.gradient(rotated)[:, ::-1],
            rtol=1e-10,
            atol=1e-14,
        ):
            raise ValueError("boundary solver requires axisymmetric faces")
        z = surface.height(xy) + surface.frame.origin[2]
        slope = surface.gradient(xy)[:, 0]
        normal = sign * np.stack((-slope, np.ones_like(r)), -1)
        normal /= np.linalg.norm(normal, axis=-1)[:, None]
        points.append(np.stack((r, z), -1))
        normals.append(normal)
        areas.append(weights * volume.rim_radius * r * np.sqrt(1 + slope**2))
    if np.any(points[1][:, 1] <= points[0][:, 1]):
        raise ValueError("lens faces intersect")
    edge = np.array([[volume.rim_radius, 0.0]])
    lower = volume.front.points(edge)[0, 2]
    upper = volume.back.points(edge)[0, 2]
    if upper <= lower:
        raise ValueError("lens must have positive rim thickness")
    points.append(
        np.stack((np.full(count, volume.rim_radius), lower + t * (upper - lower)), -1)
    )
    normals.append(np.broadcast_to([1.0, 0.0], (count, 2)))
    areas.append(weights * volume.rim_radius * (upper - lower))
    return Meridian(
        np.concatenate(points), np.concatenate(normals), np.concatenate(areas)
    )


def closed_box_grid(center, halfwidth, order=16):
    """Outward Gauss quadrature on a closed Cartesian box, lengths in metres."""
    center, halfwidth = np.asarray(center, float), np.asarray(halfwidth, float)
    if (
        center.shape != (3,)
        or halfwidth.shape != (3,)
        or not np.isfinite(center).all()
        or not np.isfinite(halfwidth).all()
        or np.any(halfwidth <= 0)
    ):
        raise ValueError("finite box center and positive halfwidths required")
    if not isinstance(order, int) or order < 2:
        raise ValueError("box quadrature order must be an integer >= 2")
    q, w = leggauss(order)
    u, v = np.meshgrid(q, q, indexing="ij")
    points, normals, weights = [], [], []
    for axis in range(3):
        tangential = [a for a in range(3) if a != axis]
        a, b = tangential
        for sign in (-1, 1):
            face = np.broadcast_to(center, (order * order, 3)).copy()
            face[:, axis] += sign * halfwidth[axis]
            face[:, a] += halfwidth[a] * u.ravel()
            face[:, b] += halfwidth[b] * v.ravel()
            points.append(face)
            normals.append(np.broadcast_to(sign * np.eye(3)[axis], face.shape))
            weights.append(
                halfwidth[a] * halfwidth[b] * (w[:, None] * w[None, :]).ravel()
            )
    return SurfaceGrid(
        np.concatenate(points), np.concatenate(normals), np.concatenate(weights)
    )
