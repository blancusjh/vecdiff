"""A beam as a Spectrum: the beam's field, referred to a chart on a plane."""

import numpy as np

from vecdiff._backend import bd
from vecdiff.fields.spectrum import Spectrum, summation
from vecdiff.fourier.polar import hankel_sum


def footprint(beam, plane):
    """Radius of the beam's geometric cone or column on a plane across its axis."""
    depth = abs(float((plane.origin - beam.frame.origin) @ beam.frame.normal))
    slope = min(float(np.tan(beam._theta_max())), 1.0)

    return float(beam._extent() + depth * slope + 10 * beam.wavelength0)


class BeamSpectrum(Spectrum):
    """A beam, the exact field of its region, referred to a chart on a plane.

    ``grid`` is the chart: a polar chart of the beam's footprint on the first
    reference plane of a meridian leg, or the lattice a lattice leg reads the
    beam on. It stands in for the lattice of a sampled input where targets
    and coverage are resolved; the beam itself is summed exactly anywhere.
    """

    def __init__(self, beam, chart, region=None):
        from vecdiff.media.absorption import BEAM, require_transparent

        self.beam, self.grid = beam, chart
        self.region = region or beam.region
        require_transparent(self.region, "a beam (fields.BeamSpectrum)", BEAM)
        self.wavelength0 = beam.wavelength0
        self.axis = beam.frame
        # One carrier object, so prepared legs are reused across calls.
        self.carrier = beam.carrier

    def relabeled(self, region):
        return BeamSpectrum(self.beam, self.grid, region)

    def _size(self):
        return f"{type(self.beam).__name__} on a {type(self.grid).__name__} chart"

    def _details(self):
        return [("beam", repr(self.beam))]

    def evaluate(
        self, grid, *, method="auto", tolerance=None, chunk_size=None, threads=None
    ):
        options = summation(method, tolerance, chunk_size, threads)
        return self.beam.evaluate(grid, **options)

    def harmonics(self, plane, radii):
        """{m: (E_rho, E_phi, E_z)} at radii on a plane across the axis, and |E|² on it.

        The azimuth is measured from the beam frame's u axis. Each order is a
        Hankel sum over the polar nodes: the azimuthal integral of
        exp(i l phi) exp(i k rho sin(theta) cos(phi - psi)) is
        2 pi i^l J_l(k rho sin(theta)) exp(i l psi).
        """
        beam = self.beam
        depth = float((plane.origin - beam.frame.origin) @ beam.frame.normal)
        reach = float(np.hypot(float(bd.max(radii)), depth))
        theta, weight, density = beam._harmonics(reach)
        k = beam.wavenumber
        sine, cosine = bd.sin(theta), bd.cos(theta)
        phase = 2 * bd.pi * weight * bd.exp(1j * k * depth * cosine)
        columns, orders = [], []

        for m, (f_theta, f_phi) in density.items():
            columns += [
                1j ** (m + 1) * phase * (cosine * f_theta + 1j * f_phi),
                1j ** (m - 1) * phase * (cosine * f_theta - 1j * f_phi),
                1j**m * phase * (-sine * f_theta),
            ]
            orders += [m + 1, m - 1, m]

        # Parseval on the plane: integral |E|² = (2 pi / k)² integral |F|² / cos dOmega.
        power = (2 * bd.pi / k) ** 2 * sum(
            2 * bd.pi * bd.sum(weight * (abs(f) ** 2 + abs(g) ** 2) / cosine)
            for f, g in density.values()
        )

        if not columns:
            return {}, 0.0

        values = hankel_sum(bd.stack(columns, -1), k * sine, radii, order=orders)
        harmonics = {}

        for index, m in enumerate(density):
            plus, minus, axial = (values[:, 3 * index + j] for j in range(3))
            harmonics[m] = ((plus + minus) / 2, (plus - minus) / 2j, axial)

        return harmonics, float(power)
