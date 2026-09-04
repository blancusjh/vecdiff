"""The seam between vecdiff's exact stigmatic solver and the general operator.

vecdiff's classic chain solves one surface — the stigmatic Cartesian oval —
essentially exactly: transfer eigenvalues on the reference spheres, then the
Debye reduction to the focal plane, the whole chain pinned against a
Franz/Stratton–Chu Maxwell reference.  The general operator of
:mod:`vecdiff.wave` handles *any* smooth surface, but it is leading order in
``1/kR``.  This module joins the two:

* :class:`OvalSurface` / :func:`oval_surface` view a
  :class:`~vecdiff.CartesianSurfaces.CartesianSurface` as a
  :class:`~vecdiff.wave.surfaces.Surface`, so the general spectral operator can
  be fed the one shape the exact solver owns;
* :func:`stigmatic_operator` builds that map with the oval's own indices and
  aperture, using the valid single-ray description of its point source;
* :func:`object_spectrum` is the matching illumination — the point source at
  the oval's object point, normalised as the exact chain normalises it;
* :func:`exact_focal_cut` drives the host package's own weighting
  (:func:`vecdiff.transfer.focal_channel_weights` + Debye reduction) for the
  same pupil;
* :func:`referee` runs both engines on the same oval and pupil and reports how
  far apart they are.

The referee is the point.  The exact solver is the trusted anchor; running the
general operator against it on the surface they share is what earns the
operator its credibility on the surfaces where no exact answer exists.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson
from scipy.special import jv

from .grids import Grid
from .operators import InterfaceOperator, point_source_spectrum
from .propagation import raised_cosine
from .spectrum import AngularSpectrum
from .surfaces import Surface

__all__ = ["OvalSurface", "oval_surface", "stigmatic_operator",
           "object_spectrum", "exact_focal_cut", "referee"]


class OvalSurface(Surface):
    """A stigmatic Cartesian oval, seen through the wave ``Surface`` interface.

    Wraps a :class:`~vecdiff.CartesianSurfaces.CartesianSurface`: ``sag`` is the
    oval's own closed-form profile (machine precision, via the host's
    ``rho -> r`` inversion), the slope is a central difference, and the usable
    radius is the host's grazing-incidence aperture limit.
    """

    def __init__(self, oval, *, step: float = 1e-6):
        self.oval = oval
        self.step = float(step)

    def sag(self, rho):
        rho = np.abs(np.asarray(rho, dtype=float))
        return np.asarray(self.oval.sag(rho), dtype=float)

    def dsag(self, rho):
        rho = np.abs(np.asarray(rho, dtype=float))
        h = self.step
        lo = np.maximum(rho - h, 0.0)
        return (self.sag(rho + h) - self.sag(lo)) / (rho + h - lo)

    @property
    def max_radius(self) -> float:
        return float(self.oval.aperture_limit)

    def __repr__(self):  # pragma: no cover - cosmetic
        o = self.oval
        return (f"OvalSurface(n0={o.n0:g}, ni={o.ni:g}, "
                f"z0={o.z0:g}, zi={o.zi:g})")


def oval_surface(oval, *, step: float = 1e-6) -> OvalSurface:
    """Return the wave :class:`Surface` view of a ``CartesianSurface``."""
    return OvalSurface(oval, step=step)


def stigmatic_operator(oval, *, wavelength: float = 1.0, mode: str = "t",
                       aperture: float | None = None,
                       **operator_kwargs) -> InterfaceOperator:
    """The general spectral operator of a stigmatic oval.

    Indices and aperture are read from the oval itself (``n1 = n0``,
    ``n2 = ni``, aperture defaulting to the grazing limit); everything else is
    forwarded to :class:`~vecdiff.wave.operators.InterfaceOperator`.  The
    matched object is a point source with one geometrical ray at every regular
    point of the oval, so ``incidence_model='local_ray'`` is the appropriate
    default here; callers may override it explicitly.
    """
    if aperture is None:
        aperture = float(oval.aperture_limit)
    operator_kwargs.setdefault("incidence_model", "local_ray")
    return InterfaceOperator(oval_surface(oval), n1=float(oval.n0),
                             n2=float(oval.ni), mode=mode, aperture=aperture,
                             wavelength=wavelength, **operator_kwargs)


def object_spectrum(oval, grid: Grid, *, wavelength: float = 1.0,
                    polarization: str = "x") -> AngularSpectrum:
    """The oval's own illumination: a point source at the object point ``A``.

    Amplitude is normalised to one at a distance ``|z0|`` from the source —
    i.e. to one on the incident reference sphere ``G`` — which is exactly the
    normalisation ``pupil(r) = 1`` means in the host's chain, so the two
    engines can be compared in absolute amplitude, not only in shape.
    """
    return point_source_spectrum(grid, distance=abs(float(oval.z0)),
                                 wavelength=wavelength, n=float(oval.n0),
                                 polarization=polarization)


def exact_focal_cut(oval, rho_obs, *, wavelength: float = 1.0,
                    pupil=None, aperture: float | None = None,
                    n_r: int = 4001, mapping: str = "sine"):
    """Focal-plane cut ``(Ex, Ez)`` along x from the host's exact chain.

    Drives :func:`vecdiff.transfer.focal_channel_weights` — the package's own
    transfer eigenvalues and pupil mapping, the ones pinned against the
    Franz/Stratton–Chu reference — followed by the Debye reduction, for an
    x-polarised pupil field ``pupil(r)`` on the incident sphere (uniform when
    ``pupil`` is None).  The observation plane is the Debye focal plane
    ``z = zi``.
    """
    from ..pupil_mapping import debye_prefactor          # noqa: PLC0415
    from ..transfer import focal_channel_weights         # noqa: PLC0415

    if aperture is None:
        aperture = float(oval.aperture_limit)
    rho_obs = np.asarray(rho_obs, dtype=float)

    r = np.linspace(0.0, float(aperture), int(n_r))
    w_p, w_s, w_z, u = focal_channel_weights(oval, r, mapping=mapping)
    f = (np.ones_like(r, dtype=complex) if pupil is None
         else np.asarray(pupil(r), dtype=complex))

    lam_plus = 0.5 * (w_p + w_s)
    lam_minus = 0.5 * (w_p - w_s)
    k_i = 2.0 * np.pi * float(oval.ni) / float(wavelength)
    q = k_i * rho_obs / abs(float(oval.zi))

    def hankel(order, g):
        integrand = g[None, :] * jv(order, q[:, None] * u[None, :]) * u[None, :]
        return simpson(integrand, x=u, axis=1)

    h0 = hankel(0, lam_plus * f)
    h2 = hankel(2, lam_minus * f)
    h1 = hankel(1, w_z * f)

    prefactor = debye_prefactor(oval, float(wavelength))
    return prefactor * (h0 - h2), 1j * prefactor * h1     # phi = 0 cut


def referee(oval, *, wavelength: float = 1.0, aperture: float | None = None,
            half_width: float = 3.0, n_obs: int = 61,
            grid: Grid | None = None, edge_softness: float = 0.25,
            **operator_kwargs) -> dict:
    """Run both engines on one oval and report how far apart they are.

    The same soft-edged pupil is applied on both sides (the general operator
    always tapers its rim; the exact chain is handed the identical taper), the
    focal-plane cut is compared at the Debye plane ``z = zi``, and the
    longitudinal channel — which comes from a different Hankel order on the
    exact side and from the surface integral on the general side — serves as an
    independent cross-check.

    Returns a dict with

    ``profile_rms``
        RMS difference of the normalised ``|Ex|`` focal profiles;
    ``fwhm_exact`` / ``fwhm_general``
        intensity FWHM of the two ``|Ex|^2`` cuts;
    ``ez_ratio_exact`` / ``ez_ratio_general``
        peak ``|Ez| / |Ex|`` of each engine;
    ``rho`` / ``Ex_exact`` / ``Ex_general`` / ``Ez_exact`` / ``Ez_general``
        the raw cuts, for plotting.
    """
    if aperture is None:
        aperture = 0.85 * float(oval.aperture_limit)
    if grid is None:
        grid = Grid.from_spacing(0.25 * float(wavelength), 256)
    rho_obs = np.linspace(0.0, half_width * float(wavelength), int(n_obs))

    def pupil(r):
        return raised_cosine(r, aperture * (1.0 - edge_softness), aperture)

    Ex_ref, Ez_ref = exact_focal_cut(oval, rho_obs, wavelength=wavelength,
                                     pupil=pupil, aperture=aperture)

    src = object_spectrum(oval, grid, wavelength=wavelength)
    op = stigmatic_operator(oval, wavelength=wavelength, aperture=aperture,
                            edge_softness=edge_softness, **operator_kwargs)
    fld = op(src).field_on(rho_obs, np.array([0.0]), z=float(oval.zi))
    Ex_gen = fld.Ex[0]
    Ez_gen = fld.components[2][0]

    p_ref = np.abs(Ex_ref) / np.abs(Ex_ref).max()
    p_gen = np.abs(Ex_gen) / np.abs(Ex_gen).max()

    def fwhm(v):
        v = v / v.max()
        above = np.flatnonzero(v >= 0.5)
        return float(2.0 * rho_obs[above[-1]]) if above.size else float("nan")

    return {
        "profile_rms": float(np.sqrt(np.mean((p_ref - p_gen) ** 2))),
        "fwhm_exact": fwhm(np.abs(Ex_ref) ** 2),
        "fwhm_general": fwhm(np.abs(Ex_gen) ** 2),
        "ez_ratio_exact": float(np.abs(Ez_ref).max() / np.abs(Ex_ref).max()),
        "ez_ratio_general": float(np.abs(Ez_gen).max() / np.abs(Ex_gen).max()),
        "rho": rho_obs,
        "Ex_exact": Ex_ref, "Ex_general": Ex_gen,
        "Ez_exact": Ez_ref, "Ez_general": Ez_gen,
    }
