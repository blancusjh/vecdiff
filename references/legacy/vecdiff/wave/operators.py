"""Composable maps on the angular spectrum.

The paper's spine: in the intrinsic representation an interface maps the space
of angular-spectrum states to itself, so interfaces *compose*.  This module
makes that literal.

* :class:`FreeSpace` -- the trivial diagonal operator (a propagation phase).
* :class:`InterfaceOperator` -- a curved dielectric interface using the local
  tangent-plane boundary model.  Fresnel transmission is applied to every
  populated incident plane-wave mode separately by default, as linearity of
  Maxwell's equations requires.  A faster one-local-ray approximation exists,
  but must be requested explicitly.
* :class:`System` -- an ordered product of maps.  Exact spectral composition is
  presently practical for sparse states; dense multi-surface examples must
  declare the local-ray approximation at the relevant interface.

Each map is callable and returns an
:class:`~vecdiff.wave.spectrum.AngularSpectrum`, so
``System([...])(spectrum)`` reads like the mathematics.  The individual-mode
spectral path is rigorous within the stated tangent-plane surface model but is
expensive for dense spectra; the explicit ``incidence_model="local_ray"`` path
is a geometrical-optics approximation and is not a linear Maxwell operator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .grids import Grid
from .interfaces import reflect_field, transmit_field
from .propagation import (_axisym_samples, _freeform_samples,
                          _rim_tir_apodization, _return_integral_nufft,
                          _return_integral_polar, incident_on_surface,
                          raised_cosine, spectrum_of)
from .spectrum import AngularSpectrum
from .surfaces import Surface

__all__ = ["Operator", "FreeSpace", "InterfaceOperator", "System",
           "plane_wave_spectrum", "point_source_spectrum"]


class Operator(ABC):
    """A map from an angular spectrum to an angular spectrum."""

    @abstractmethod
    def apply(self, spec: AngularSpectrum) -> AngularSpectrum:
        ...

    def __call__(self, spec: AngularSpectrum) -> AngularSpectrum:
        return self.apply(spec)


# --------------------------------------------------------------------------
class FreeSpace(Operator):
    """Propagation through a homogeneous gap of length ``distance``.

    Diagonal on the angular spectrum: every direction is multiplied by its own
    phase and none is mixed.  The index is inherited from the incoming
    spectrum (free space does not change the medium)."""

    def __init__(self, distance: float):
        self.distance = float(distance)

    def apply(self, spec: AngularSpectrum) -> AngularSpectrum:
        mask = spec.grid.propagating(spec.k)
        phase = mask * np.exp(1j * spec.sigma * spec.kz * self.distance)
        return AngularSpectrum(spec.A * phase[None], spec.grid, spec.wavelength,
                               spec.n, spec.sigma)

    def __repr__(self):  # pragma: no cover - cosmetic
        return f"FreeSpace({self.distance:g})"


# --------------------------------------------------------------------------
class InterfaceOperator(Operator):
    """A curved dielectric interface, as an operator on the angular spectrum.

    ``apply`` samples every populated incident plane-wave mode on the surface,
    refracts (``mode='t'``) or reflects (``mode='r'``) it with its own full
    vector Fresnel operator, and sums the outgoing spectra.  This mode-by-mode
    action preserves superposition exactly.  For a surface of revolution the
    azimuthal Bessel kernel is used
    (``method='polar'``); a :class:`~vecdiff.wave.surfaces.Freeform2D` surface, or
    ``method='nufft'``, uses the general NUFFT transform.

    ``incidence_model="spectral"`` is the physically safe default.  It is
    intentionally capped by ``max_spectral_modes`` because its present direct
    implementation performs one surface transform per populated mode.  The
    alternative ``incidence_model="local_ray"`` reconstructs one energy-flow
    direction from the total field at each surface point.  That approximation
    is appropriate only when one geometrical ray reaches each point (for
    example, a point source in its single-ray region); for interfering fields
    it is nonlinear and must never be mistaken for the general Maxwell map.
    """

    def __init__(self, surface: Surface, *, n1: float, n2: float,
                 mode: str = "t", aperture: float | None = None,
                 wavelength: float = 1.0, m_max: int = 6,
                 n_rho: int = 600, n_phi: int = 64, n_kr: int = 512,
                 n_free: int = 220, edge_softness: float = 0.25,
                 tir_margin: float = 0.04, method: str = "auto",
                 measure: str = "franz", incidence_model: str = "spectral",
                 max_spectral_modes: int = 64):
        self.surface = surface
        self.n1, self.n2, self.mode = float(n1), float(n2), mode
        self.wavelength = float(wavelength)
        self.m_max, self.n_rho, self.n_phi = m_max, n_rho, n_phi
        self.n_kr, self.n_free = n_kr, n_free
        self.edge_softness, self.tir_margin = edge_softness, tir_margin
        self.method = method
        if incidence_model not in ("spectral", "local_ray"):
            raise ValueError("incidence_model must be 'spectral' or 'local_ray'")
        if int(max_spectral_modes) < 1:
            raise ValueError("max_spectral_modes must be positive")
        self.incidence_model = incidence_model
        self.max_spectral_modes = int(max_spectral_modes)
        if measure not in ("franz", "flat"):
            raise ValueError("measure must be 'franz' or 'flat'")
        self.measure = measure
        if aperture is None:
            aperture = (surface.max_radius if np.isfinite(surface.max_radius)
                        else 10.0 * self.wavelength)
        self.aperture = float(aperture)

    @property
    def out_index(self) -> float:
        return self.n2 if self.mode == "t" else self.n1

    def _local(self, points, nhat, khat, E_in):
        if self.mode == "t":
            return transmit_field(E_in, khat, nhat, self.n1, self.n2)
        return reflect_field(E_in, khat, nhat, self.n1, self.n2)

    def _franz_pieces(self, nhat, k_out_dir):
        """Oriented normal and the Q-side obliquity half ``(n.k_out)/2``.

        The outward normal of the radiating half-space is ``nhat`` for
        transmission and ``-nhat`` for reflection.  The latter accompanies a
        negative output ``sigma``.  Returns ``(None, None)`` for the deliberately
        unweighted ``flat`` measure.
        """
        if self.measure != "franz":
            return None, None
        outward = nhat if self.mode == "t" else -nhat
        pair = 0.5 * np.abs(np.sum(outward * k_out_dir, axis=0))
        return outward, pair

    def _apply_local_ray(self, spec: AngularSpectrum) -> AngularSpectrum:
        """Apply the single-local-ray construction once to the total field."""
        k_out = 2 * np.pi * self.out_index / self.wavelength
        out_sigma = spec.sigma if self.mode == "t" else -spec.sigma
        axisym = getattr(self.surface, "rotationally_symmetric", True)
        method = self.method
        if method == "auto":
            method = "polar" if axisym else "nufft"
        if not axisym and method == "polar":
            raise ValueError("polar kernel needs an axisymmetric surface")

        if axisym:
            smp = _axisym_samples(self.surface, self.aperture, self.n_rho, self.n_phi)
            pts, nhat = smp["points"], smp["nhat"]
            E_in, khat = incident_on_surface(spec, pts[0], pts[1], pts[2])
            E_out, k_dir, coeffs = self._local(pts, nhat, khat, E_in)
            nhat_out, pair = self._franz_pieces(nhat, k_dir)
            rho, phi = smp["rho"], smp["phi"]
            vis = _rim_tir_apodization(rho, len(phi), self.aperture,
                                       self.edge_softness, coeffs, self.mode,
                                       self.n1, self.n2, self.tir_margin)
            visg = np.broadcast_to(vis[:, None], smp["RHO"].shape).ravel()
            E_out = E_out * visg[None, :]
            if method == "polar":
                shape3 = (3, len(rho), len(phi))
                datum = E_out.reshape(shape3)
                datum_pair = None
                normal_rz = None
                if pair is not None:
                    datum_pair = (E_out * pair[None, :]).reshape(shape3)
                    normal_rz = self.surface.normal(rho)
                    if self.mode == "r":
                        normal_rz = tuple(-component for component in normal_rz)
                return _return_integral_polar(datum, rho, smp["sag"],
                                              self.surface.dsag(rho), k_out,
                                              spec.grid, self.m_max, self.n_kr,
                                              out_sigma, self.wavelength,
                                              self.out_index,
                                              datum_pair=datum_pair,
                                              normal_rz=normal_rz)
            drho, dphi = rho[1] - rho[0], 2 * np.pi / len(phi)
            area = (rho * np.sqrt(1.0 + self.surface.dsag(rho) ** 2) * drho * dphi)
            dS = np.broadcast_to(area[:, None], smp["RHO"].shape).ravel()
            return _return_integral_nufft(pts, E_out, dS, k_out, spec.grid,
                                          out_sigma, self.wavelength,
                                          self.out_index,
                                          pair_weight=pair, nhat=nhat_out)

        # ---- general freeform surface -----------------------------------
        smp = _freeform_samples(self.surface, self.aperture, self.n_free)
        pts, nhat, dS = smp["points"], smp["nhat"], smp["dS"]
        E_in, khat = incident_on_surface(spec, pts[0], pts[1], pts[2])
        E_out, k_dir, coeffs = self._local(pts, nhat, khat, E_in)
        nhat_out, pair = self._franz_pieces(nhat, k_dir)
        r = np.hypot(pts[0], pts[1])
        vis = raised_cosine(r, self.aperture * (1.0 - self.edge_softness),
                            self.aperture)
        E_out = E_out * vis[None, :]
        return _return_integral_nufft(pts, E_out, dS * vis, k_out, spec.grid,
                                      out_sigma, self.wavelength,
                                      self.out_index,
                                      pair_weight=pair, nhat=nhat_out)

    def _apply_spectral(self, spec: AngularSpectrum) -> AngularSpectrum:
        """Apply Fresnel mode by mode and sum, preserving superposition."""
        populated = (np.any(spec.A != 0.0, axis=0)
                     & spec.grid.propagating(spec.k))
        modes = np.argwhere(populated)
        if len(modes) == 0:
            sigma = spec.sigma if self.mode == "t" else -spec.sigma
            return AngularSpectrum(np.zeros_like(spec.A), spec.grid,
                                   spec.wavelength, self.out_index, sigma)
        if len(modes) > self.max_spectral_modes:
            raise ValueError(
                f"spectral incidence has {len(modes)} populated modes, above "
                f"max_spectral_modes={self.max_spectral_modes}; increase the "
                "limit for the exact (but expensive) mode-by-mode map, or "
                "request incidence_model='local_ray' explicitly only when one "
                "geometrical ray reaches each surface point"
            )

        total = None
        template = np.zeros_like(spec.A)
        for iy, ix in modes:
            amplitude = template.copy()
            amplitude[:, iy, ix] = spec.A[:, iy, ix]
            incident_mode = AngularSpectrum(amplitude, spec.grid,
                                            spec.wavelength, spec.n,
                                            spec.sigma)
            outgoing_mode = self._apply_local_ray(incident_mode)
            if total is None:
                total = np.zeros_like(outgoing_mode.A)
                out_sigma = outgoing_mode.sigma
            total += outgoing_mode.A
        return AngularSpectrum(total, spec.grid, spec.wavelength,
                               self.out_index, out_sigma)

    def apply(self, spec: AngularSpectrum) -> AngularSpectrum:
        if not np.isclose(spec.n, self.n1, rtol=1e-12, atol=1e-14):
            raise ValueError(
                f"incident spectrum index n={spec.n:g} does not match "
                f"interface incident index n1={self.n1:g}"
            )
        if not np.isclose(spec.wavelength, self.wavelength,
                          rtol=1e-12, atol=1e-14):
            raise ValueError(
                f"incident wavelength {spec.wavelength:g} does not match "
                f"operator wavelength {self.wavelength:g}"
            )
        if self.incidence_model == "local_ray":
            return self._apply_local_ray(spec)
        return self._apply_spectral(spec)

    def __repr__(self):  # pragma: no cover - cosmetic
        return (f"InterfaceOperator({type(self.surface).__name__}, "
                f"n1={self.n1:g}, n2={self.n2:g}, mode={self.mode!r}, "
                f"incidence_model={self.incidence_model!r})")


# --------------------------------------------------------------------------
class System(Operator):
    """An ordered product of operators: ``System([A, B, C])`` applies A, then B,
    then C.  The algebraic composition is exact.  A dense spectrum at a later
    interface may require either an expensive spectral evaluation or an
    explicitly selected ``local_ray`` approximation; :class:`InterfaceOperator`
    never changes between those regimes silently."""

    def __init__(self, operators):
        self.operators = list(operators)

    def apply(self, spec: AngularSpectrum) -> AngularSpectrum:
        for op in self.operators:
            spec = op.apply(spec)
        return spec

    def __mul__(self, other: "Operator") -> "System":
        """``self * other`` = apply ``other`` first, then ``self`` (operator order)."""
        right = other.operators if isinstance(other, System) else [other]
        return System(list(right) + self.operators)

    def __len__(self):
        return len(self.operators)

    def __repr__(self):  # pragma: no cover - cosmetic
        return "System([" + ", ".join(repr(o) for o in self.operators) + "])"


# --------------------------------------------------------------------------
#  Source spectra to drive a system
# --------------------------------------------------------------------------
def plane_wave_spectrum(grid: Grid, *, wavelength: float = 1.0, n: float = 1.0,
                        polarization: str = "x", amplitude: float = 1.0,
                        sigma: int = +1) -> AngularSpectrum:
    """A unit on-axis plane wave as an angular spectrum (a single k=0 mode)."""
    from .pupil import POLARIZATIONS
    pol = polarization if callable(polarization) else POLARIZATIONS[polarization]
    ex, ey = pol(np.zeros(()), np.zeros(()))
    area = (grid.x.size * grid.dx) * (grid.y.size * grid.dy)
    A = np.zeros((3, *grid.shape), dtype=complex)
    A[0, 0, 0] = amplitude * complex(ex) * area
    A[1, 0, 0] = amplitude * complex(ey) * area
    return AngularSpectrum(A, grid, wavelength, n, sigma).impose_transversality()


def point_source_spectrum(grid: Grid, *, distance: float, wavelength: float = 1.0,
                          n: float = 1.0, polarization: str = "x",
                          sigma: int = +1) -> AngularSpectrum:
    """A diverging spherical wave from a point a distance ``distance`` before the
    plane ``z = 0``, reduced to its angular spectrum."""
    from .fields import Field
    from .pupil import POLARIZATIONS
    pol = polarization if callable(polarization) else POLARIZATIONS[polarization]
    X, Y = grid.XY
    R = np.hypot(np.hypot(X, Y), distance)
    k = 2 * np.pi * n / wavelength
    env = np.exp(1j * k * R) * (distance / R)
    ex, ey = pol(grid.R / max(grid.R.max(), 1e-12), grid.PHI)
    field = Field(np.asarray(ex) * env, np.asarray(ey) * env, grid, wavelength, n)
    return spectrum_of(field, sigma).impose_transversality()
