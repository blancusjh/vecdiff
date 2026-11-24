"""Vectorial refraction and focusing by a real high-NA lens: the README's figures.

The lens is stigmatic: an ellipsoidal entry face (eccentricity 1/n) sends a
collimated beam to its far focus inside the glass, and an exit sphere centred
on that focus lets every ray out at normal incidence. Glass n = 1.5,
semi-axis 3 mm, focus 5 mm behind the vertex, stop cut at NA = 0.7 (4.45 mm
across, 8400 wavelengths at 532 nm); the marginal ray meets the ellipsoid at
86 degrees of incidence.

A second lens is aplanatic as well as stigmatic: two aspheric faces computed
by the method of Wassermann and Wolf (Proc. Phys. Soc. B 62, 2, 1949) so that
every ray reaches the focus and leaves at sin(theta) = h / f, the sine
condition of the ideal lens. Both are compared with that ideal lens.

Run as a script it prints the system, the report and the focal numbers of
four input polarizations, with two independent checks (about 20 s). With a
directory as argument it writes the README's figures there; the map through
the whole lens takes a few minutes. Name figures after the directory to write
only those: ``meridional``, ``refraction``, ``focal``, ``lenses``, ``mie``.
"""

import sys
from pathlib import Path

import numpy as np
import vecdiff as vd

WAVELENGTH = 532e-9
INDEX, SEMI_AXIS, NA = 1.5, 3e-3, 0.7
ECCENTRICITY = 1 / INDEX
FOCUS = SEMI_AXIS * (1 + ECCENTRICITY)
EXIT_RADIUS = 2.4e-3
Frame = vd.geometry.Frame


def ellipsoid(z):
    """Height of the entry face at depth z behind its vertex."""
    minor = SEMI_AXIS * np.sqrt(1 - ECCENTRICITY**2)
    return minor * np.sqrt(1 - ((z - SEMI_AXIS) / SEMI_AXIS) ** 2)


def marginal():
    """Height and depth at which the ray of sin(theta) = NA meets the entry face."""
    z = np.linspace(0, SEMI_AXIS, 300001)
    h = ellipsoid(z)
    k = np.searchsorted(h / np.hypot(h, FOCUS - z), NA)
    return float(h[k]), float(z[k])


APERTURE, EDGE_DEPTH = marginal()


def lens():
    """The path: the stop, the ellipsoid, the sphere centred on the focus."""
    air, glass = vd.media.air, vd.media.Dielectric(INDEX, "glass")
    entry = vd.surfaces.CartesianOval.from_conjugates(1.0, -np.inf, INDEX, FOCUS)
    exit_face = vd.surfaces.SphericalCap(
        EXIT_RADIUS, Frame([0, 0, FOCUS - EXIT_RADIUS])
    )
    return vd.system.OpticalSystem(
        vd.system.Stop(APERTURE, z=-0.2e-3),
        vd.system.Refraction(entry, air, glass),
        vd.system.Refraction(exit_face, glass, air),
    )


APLANAT_APERTURE, APLANAT_THICKNESS, APLANAT_BACK = 2.2e-3, 3.2e-3, 1.4e-3
APLANAT_FOCAL = APLANAT_APERTURE / NA
APLANAT_FOCUS = APLANAT_THICKNESS + APLANAT_BACK


def aplanat_rays(count=3001, margin=1.03):
    """The two faces of the aplanatic lens, ray by ray (Wassermann and Wolf).

    The ray entering at height h must leave the second face toward the focus
    at sin(theta) = h / f with the optical path of the axial ray. That fixes
    where it meets the second face once the first is known, and Snell's law
    on the first face then gives its slope: an ordinary differential
    equation for the first face, integrated from the vertex. Returns, per
    ray: h, the first face's sag and slope, the second face's radius, sag and
    slope, and the cosines of incidence and refraction on both.
    """
    from scipy.integrate import solve_ivp

    n, f, focus = INDEX, APLANAT_FOCAL, APLANAT_FOCUS
    path = n * APLANAT_THICKNESS + APLANAT_BACK

    def ray(h, z1):
        sine = h / f
        cosine = np.sqrt(1 - sine**2)
        depth, rest = focus - z1, path - z1
        a = n**2 - 1
        b = n**2 * (sine * h + cosine * depth) - rest
        c = n**2 * (h**2 + depth**2) - rest**2
        t = (b - np.sqrt(b**2 - a * c)) / a  # length of the last segment
        r2, z2 = t * sine, focus - t * cosine
        d = np.hypot(r2 - h, z2 - z1)
        return r2, z2, (r2 - h) / d, (z2 - z1) / d, sine, cosine

    def slope(h, z1):
        _, _, ur, uz, _, _ = ray(h, z1)
        return n * ur / (1 - n * uz)

    h = np.linspace(0, margin * APLANAT_APERTURE, count)
    z1 = solve_ivp(
        lambda h, y: [slope(h, y[0])],
        (0, h[-1]),
        [0.0],
        t_eval=h,
        rtol=1e-12,
        atol=1e-15,
        method="DOP853",
    ).y[0]
    slope1 = slope(h, z1)
    r2, z2, ur, uz, sine, cosine = ray(h, z1)
    # The second face is normal to n u - v, v the direction toward the focus.
    slope2 = -(n * ur + sine) / (n * uz - cosine)
    cos_i1 = 1 / np.hypot(slope1, 1)
    cos_i2 = np.abs(uz - slope2 * ur) / np.hypot(slope2, 1)
    cosines = (
        cos_i1,
        np.sqrt(1 - (1 - cos_i1**2) / n**2),
        cos_i2,
        np.sqrt(np.clip(1 - n**2 * (1 - cos_i2**2), 0, 1)),
    )
    return h, z1, slope1, r2, z2, slope2, cosines


def aplanat():
    """The path through the aplanatic lens: the stop and its two aspheric faces."""
    from scipy.interpolate import make_interp_spline

    h, z1, slope1, r2, z2, slope2, _ = aplanat_rays()

    def face(radius, sag, slope, vertex):
        rise = make_interp_spline(radius, slope, k=5)
        return vd.surfaces.AxisymmetricSurface(
            make_interp_spline(radius, sag - vertex, k=5),
            rise,
            Frame([0, 0, vertex]),
            rise.derivative(),
        )

    air, glass = vd.media.air, vd.media.Dielectric(INDEX, "glass")
    return vd.system.OpticalSystem(
        vd.system.Stop(APLANAT_APERTURE, z=-0.2e-3),
        vd.system.Refraction(face(h, z1, slope1, 0.0), air, glass),
        vd.system.Refraction(face(r2, z2, slope2, APLANAT_THICKNESS), glass, air),
    )


def aplanat_focus(half_width=1.6e-6, count=161):
    """The linear-x beam through the aplanatic lens, on its focal plane."""
    light = vd.fields.PlaneBeam(
        vd.fields.profiles.Gaussian(2 * APLANAT_APERTURE),
        vd.fields.polarization.x,
        wavelength0=WAVELENGTH,
        region=vd.media.air,
        frame=Frame([0, 0, -0.4e-3]),
    )
    solution = vd.propagation.Propagator(aplanat())(light)
    window = (-half_width, half_width, -half_width, half_width)
    return solution.field(z=APLANAT_FOCUS, window=window, shape=(count, count))


def debye_aplanat():
    """Peak |E_x|, |E_y|, |E_z| of the aplanatic lens's focus from ray optics.

    Each ray carries the s and p Fresnel coefficients of both faces with
    their flux factors, and the aplanatic factor f sqrt(cos theta); the focus
    is the Debye sum, in the three integrals of Richards and Wolf.
    """
    from scipy.special import jv

    n = INDEX
    h, _, _, _, _, _, (ci1, ct1, ci2, ct2) = aplanat_rays(20001, margin=1.0)
    h, ci1, ct1, ci2, ct2 = (v[1:] for v in (h, ci1, ct1, ci2, ct2))
    theta = np.arcsin(h / APLANAT_FOCAL)

    def carried(ni, nt, ci, ct):
        flux = np.sqrt(nt * ct / (ni * ci))
        return flux * 2 * ni * ci / (ni * ci + nt * ct), flux * 2 * ni * ci / (
            nt * ci + ni * ct
        )

    s1, p1 = carried(1, n, ci1, ct1)
    s2, p2 = carried(n, 1, ci2, ct2)
    pupil = np.exp(-((h / (2 * APLANAT_APERTURE)) ** 2))
    common = pupil * APLANAT_FOCAL * np.sqrt(np.cos(theta)) * np.sin(theta)
    a_s, a_p = common * s1 * s2, common * p1 * p2
    k = 2 * np.pi / WAVELENGTH
    phase = k * np.outer(np.linspace(0, 1.6e-6, 400), np.sin(theta))
    i0 = np.trapezoid((a_s + a_p * np.cos(theta)) * jv(0, phase), theta, axis=1)
    i1 = np.trapezoid(a_p * np.sin(theta) * jv(1, phase), theta, axis=1)
    i2 = np.trapezoid((a_s - a_p * np.cos(theta)) * jv(2, phase), theta, axis=1)
    return k / 2 * abs(i0[0]), k / 2 * np.abs(i2).max(), k * np.abs(i1).max()


POLARIZATIONS = {
    "linear x": vd.fields.polarization.x,
    "circular": vd.fields.polarization.cartesian(1, 1j),
    "radial": vd.fields.polarization.radial,
    "azimuthal": vd.fields.polarization.azimuthal,
}


def beam(polarization):
    """A collimated Gaussian beam twice as wide as the stop, 1 V/m at its centre."""
    return vd.fields.PlaneBeam(
        vd.fields.profiles.Gaussian(2 * APERTURE),
        polarization,
        wavelength0=WAVELENGTH,
        region=vd.media.air,
        frame=Frame([0, 0, -0.4e-3]),
    )


def solutions():
    """Each input polarization carried through the lens."""
    propagator = vd.propagation.Propagator(lens())
    return {name: propagator(beam(state)) for name, state in POLARIZATIONS.items()}


def focal_plane(solution, half_width=1.6e-6, count=161):
    window = (-half_width, half_width, -half_width, half_width)
    return solution.field(z=FOCUS, window=window, shape=(count, count))


def ideal_focus(grid, aperture=APERTURE):
    """The same pupil focused by an ideal aplanatic lens (Richards and Wolf)."""
    ideal = vd.fields.FocusedBeam(
        vd.fields.profiles.Gaussian(2 * aperture),
        vd.fields.polarization.x,
        wavelength0=WAVELENGTH,
        region=vd.media.air,
        focal_length=aperture / NA,
        max_sine=NA,
        frame=grid.frame,
    )
    return ideal.evaluate(grid)


def debye_axis_amplitude():
    """|E_x| at the focus from ray optics alone, for the linear-x beam.

    Each ray carries t_s of the ellipsoid (the transverse field of a
    collimated beam stays linear across one refraction, since
    t_p / t_s = 1 / cos(theta_i - theta_t)), the tube factor of the
    refraction and of the map h -> theta, and the normal-incidence
    coefficient of the exit sphere; the focus is their Debye sum.
    """
    z = np.linspace(0, EDGE_DEPTH, 400001)[1:]
    h = ellipsoid(z)
    theta = np.arctan2(h, FOCUS - z)
    refracted = np.arctan2(np.sin(theta), INDEX - np.cos(theta))
    incident = refracted + theta
    t_s = 2 * np.cos(incident) / (np.cos(incident) + INDEX * np.cos(refracted))
    tube = np.cos(refracted) / np.cos(incident) * h * np.gradient(h, theta)
    amplitude = (
        t_s
        * (2 * INDEX / (INDEX + 1))
        * np.exp(-((h / (2 * APERTURE)) ** 2))
        * np.sqrt(tube / np.sin(theta))
    )
    wavenumber = 2 * np.pi / WAVELENGTH
    return float(wavenumber * np.trapezoid(amplitude * np.sin(theta), theta))


def outlines(axis, half_height):
    """Draw the stop and the lens on a (z, x) axis in millimetres."""
    z = np.linspace(0, EDGE_DEPTH, 400)
    front = ellipsoid(z)
    x = np.linspace(-APERTURE, APERTURE, 400)
    back = FOCUS - np.sqrt(EXIT_RADIUS**2 - x**2)
    style = {"color": "white", "lw": 1.0}
    for sign in (-1, 1):
        axis.plot(z * 1e3, sign * front * 1e3, **style)
        axis.plot(
            [EDGE_DEPTH * 1e3, back[0] * 1e3],
            [sign * APERTURE * 1e3] * 2,
            **style,
        )
        axis.plot(
            [-0.2, -0.2],
            [sign * APERTURE * 1e3, sign * half_height * 1e3],
            color="white",
            lw=2.5,
        )
    axis.plot(back * 1e3, x * 1e3, **style)


def figure_meridional(carried, path):
    """|E|² through the lens, and the polarization in the plane of propagation."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from vecdiff.visualization import plot_polarization

    fig = plt.figure(figsize=(15.5, 6.6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.45, 1])
    wide = fig.add_subplot(grid[:, 0])
    half, frame = 2.23e-3, 2.55e-3
    solution = carried["linear x"]
    field = solution.field(
        y=0.0, window=(-half, half, -0.35e-3, FOCUS + 0.8e-3), shape=(440, 301)
    )
    intensity = np.sum(np.abs(field.electric.values) ** 2, axis=-1).T
    image = wide.imshow(
        intensity,
        extent=(-0.35, (FOCUS + 0.8e-3) * 1e3, -half * 1e3, half * 1e3),
        origin="lower",
        aspect="equal",
        cmap="inferno",
        norm=LogNorm(0.1, 1e3),
    )
    outlines(wide, frame)
    wide.set_facecolor("black")
    wide.set(xlabel="z (mm)", ylabel="x (mm)", ylim=(-frame * 1e3, frame * 1e3))
    wide.set_title("|E|² from the stop, through the glass, to the focus", loc="left")
    fig.colorbar(
        image,
        ax=wide,
        pad=0.01,
        fraction=0.04,
        extend="max",
        label=r"$|\mathbf{E}|^2$ (V²/m²), incident beam = 1",
    )

    # The plane of propagation around the focus, z to the right as above.
    width, depth = 1.2e-6, 1.7e-6
    plane = Frame([0, 0, FOCUS], [[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    target = vd.sampling.CartesianGrid(
        np.linspace(-depth, depth, 171), np.linspace(-width, width, 121), plane
    )
    for row, name in enumerate(("linear x", "radial")):
        axis = fig.add_subplot(grid[row, 1])
        section = carried[name].field(target)
        values = section.electric.values
        peak = float(np.sum(np.abs(values) ** 2, axis=-1).max())
        plot_polarization(
            section.electric,
            axis=axis,
            crop=False,
            legend=False,
            title=f"{name} input: E in the plane of propagation, at the focus",
            norm=LogNorm(1e-3, 1),
            reference_intensity=peak,
            colorbar_label=r"$(|E_x|^2+|E_z|^2)$ / peak",
        )
        axis.set(xlabel="z − focus (µm)", ylabel="x (µm)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_refraction(solution, path):
    """The converging beam 1 mm before focus: what one refraction did to it."""
    import matplotlib.pyplot as plt

    half = 1.15e-3
    field = solution.field(
        z=FOCUS - 1e-3, window=(-half, half, -half, half), shape=(241, 241)
    )
    values = field.electric.values
    peak = float((np.abs(values[..., 0]) ** 2).max())
    ceiling = float((np.abs(values[..., 2]) ** 2).max()) / peak
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
    extent = (-half * 1e3, half * 1e3, -half * 1e3, half * 1e3)
    for axis, component, label in zip(axes, range(3), "xyz"):
        share = np.abs(values[..., component]) ** 2 / peak
        image = axis.imshow(
            share,
            extent=extent,
            origin="lower",
            cmap="inferno",
            vmin=0,
            vmax=1 if component == 0 else ceiling,
        )
        axis.set_title(rf"$|E_{label}|^2$ (peak {share.max():.1e} of $|E_x|^2$)")
        axis.set(xlabel="x (mm)", ylabel="y (mm)")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    centre = values.shape[0] // 2
    x = field.grid.u * 1e3
    sine = x / np.hypot(x, 1.0)
    axes[3].plot(sine, np.abs(values[centre, :, 0]), label=r"$|E_x|$ along x (p side)")
    axes[3].plot(
        sine, np.abs(values[:, centre, 0]), "--", label=r"$|E_x|$ along y (s side)"
    )
    axes[3].plot(sine, np.abs(values[centre, :, 2]), label=r"$|E_z|$ along x")
    axes[3].set(xlabel=r"$\sin\theta$ of the ray", ylabel="V/m", xlim=(-0.75, 0.75))
    axes[3].set_title("Meridians of the converging beam")
    axes[3].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_focal(fields, path):
    """Polarization over transverse intensity, and |E_z|², for each input."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from vecdiff.visualization import plot_polarization

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.4))
    for (name, field), top, bottom in zip(fields.items(), *axes):
        values = field.electric.values
        total = float(np.sum(np.abs(values) ** 2, axis=-1).max())
        plot_polarization(
            field.electric,
            axis=top,
            crop=False,
            legend=False,
            title=f"{name} input: transverse field",
            norm=LogNorm(1e-3, 1),
            reference_intensity=total,
            colorbar_label=r"$(|E_x|^2+|E_y|^2)$ / peak $|\mathbf{E}|^2$",
        )
        top.set(xlabel="x (µm)", ylabel="y (µm)")
        u, v = field.grid.u * 1e6, field.grid.v * 1e6
        longitudinal = np.abs(values[..., 2]) ** 2 / total
        image = bottom.imshow(
            np.maximum(longitudinal, 1e-12),
            extent=(u[0], u[-1], v[0], v[-1]),
            origin="lower",
            cmap="inferno",
            norm=LogNorm(1e-3, 1),
        )
        share = float(longitudinal.max())
        label = f"peak {share:.2f}" if share > 1e-12 else "zero"
        bottom.set_title(rf"$|E_z|^2$ / peak $|\mathbf{{E}}|^2$ ({label})")
        bottom.set(xlabel="x (µm)", ylabel="y (µm)")
        fig.colorbar(image, ax=bottom, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_lenses(stigmatic, aplanatic, path):
    """Two real lenses against the ideal aplanatic lens, component by component."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    ideal = ideal_focus(aplanatic.grid, APLANAT_APERTURE)
    fig, axes = plt.subplots(3, 4, figsize=(17, 12.2))
    u = stigmatic.grid.u * 1e6
    extent = (u[0], u[-1], u[0], u[-1])
    centre = len(u) // 2
    rows = (
        ("stigmatic lens", stigmatic),
        ("aplanatic lens", aplanatic),
        ("ideal lens", ideal),
    )
    for row, (name, field) in zip(axes, rows):
        values = field.electric.values
        peak = float((np.abs(values[..., 0]) ** 2).max())
        for axis, component, label in zip(row, range(3), "xyz"):
            share = np.abs(values[..., component]) ** 2 / peak
            image = axis.imshow(
                np.maximum(share, 1e-12),
                extent=extent,
                origin="lower",
                cmap="inferno",
                norm=LogNorm(1e-5, 1),
            )
            axis.set_title(rf"{name}: $|E_{label}|^2$, peak {share.max():.1e}")
            axis.set(xlabel="x (µm)", ylabel="y (µm)")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        total = np.sum(np.abs(values) ** 2, axis=-1)
        total = total / total.max()
        row[3].semilogy(u, total[centre], label="along x")
        row[3].semilogy(u, total[:, centre], "--", label="along y")
        row[3].set(xlabel="µm", ylabel=r"$|\mathbf{E}|^2$ / peak", ylim=(1e-5, 1.5))
        row[3].set_title(rf"{name}: total $|\mathbf{{E}}|^2$")
        row[3].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def weierstrass(radius):
    """A sphere used at its aplanatic points, against the exact Mie solution.

    A beam converging toward A = n R behind the centre of a glass sphere is
    focused without aberration at B = R / n (Young and Weierstrass). Lengths
    in wavelengths. Returns the focal radii, the carried field, the exact
    first-entry (Debye p = 1) field and the full Mie field.
    """
    from vecdiff.references import MieSphere

    air = vd.media.HomogeneousRegion("air", vd.media.air)
    glass = vd.media.HomogeneousRegion("glass", vd.media.Dielectric(INDEX, "glass"))
    u_max = 0.4
    incoming = vd.fields.FocusedBeam(
        lambda x, y: np.clip(1 - (x**2 + y**2) / u_max**2, 0, None) ** 3,
        vd.fields.polarization.cartesian(1 / np.sqrt(2), 1j / np.sqrt(2)),
        focal_length=1.0,
        max_sine=u_max,
        wavelength0=1.0,
        region=air,
        frame=Frame([0, 0, INDEX * radius]),
        apodization="none",
    )
    face = vd.surfaces.SphericalCap(radius, Frame([0, 0, -radius]))
    propagator = vd.propagation.Propagator(
        (vd.system.Refraction(face, air, glass),), planes=(0.0,)
    )
    rho = np.linspace(0, 5, 81)
    points = vd.sampling.PointGrid(np.c_[rho, 0 * rho, np.full(81, radius / INDEX)])
    carried = propagator(incoming).spectrum.evaluate(points).electric.values
    mie = MieSphere(incoming, vd.volumes.SphereVolume([0, 0, 0], radius), glass)
    first = mie.evaluate(points, term=1, side="interior").electric.values
    full = mie.evaluate(points, side="interior").electric.values
    return rho, carried, first, full


def misfit(rho, field, reference):
    """Relative L2 difference of two complex fields over the focal disk."""
    from scipy.integrate import simpson

    def norm(values):
        return simpson(rho * np.sum(np.abs(values) ** 2, axis=-1), x=rho)

    return float(np.sqrt(norm(field - reference) / norm(reference)))


def figure_mie(path, radii=(100.0, 300.0, 1000.0, 3000.0)):
    """The aplanatic sphere against Mie: the fields, and the error against size."""
    import matplotlib.pyplot as plt

    fig, (profile, trend) = plt.subplots(1, 2, figsize=(12, 4.6))
    errors = []
    for radius in radii:
        rho, carried, first, full = weierstrass(radius)
        errors.append(misfit(rho, carried, first))
        if radius == 1000.0:
            for component, label in ((0, "x"), (2, "z")):
                profile.semilogy(
                    rho, np.abs(first[:, component]), lw=3, alpha=0.35, color="k"
                )
                profile.semilogy(
                    rho, np.abs(carried[:, component]), label=rf"$|E_{label}|$"
                )
            profile.set(
                xlabel="distance from the axis (wavelengths)",
                ylabel="relative units",
                ylim=(1e-7, 1),
            )
            profile.set_title(
                "Sphere R = 1000 λ at its aplanatic points:\n"
                "vecdiff (lines) on the exact Mie field (grey)"
            )
            profile.legend()
    trend.loglog(radii, errors, "o-", label="vecdiff against Mie")
    trend.loglog(
        radii, errors[0] * radii[0] / np.asarray(radii), "k:", label="1 / size"
    )
    trend.set(xlabel="sphere radius (wavelengths)", ylabel="relative L2 error")
    trend.set_title("The error falls as the system grows")
    trend.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return dict(zip(radii, errors))


def main(figures=None, *only):
    print(lens())
    carried = solutions()
    print(carried["linear x"])
    fields = {name: focal_plane(solution) for name, solution in carried.items()}
    for name, field in fields.items():
        peak = np.abs(field.electric.values).max(axis=(0, 1))
        print(
            f"{name:>10}: max |Ex|, |Ey|, |Ez| = "
            f"{peak[0]:7.1f}, {peak[1]:9.2e}, {peak[2]:7.1f} V/m"
        )
    centre = fields["linear x"].grid.shape[0] // 2
    on_axis = abs(fields["linear x"].electric.values[centre, centre, 0])
    print(
        f"|Ex| at the focus: {on_axis:.1f} V/m; ray-optics Debye sum: "
        f"{debye_axis_amplitude():.1f} V/m"
    )
    aplanatic = aplanat_focus()
    rows = (
        ("stigmatic lens", fields["linear x"], APERTURE),
        ("aplanatic lens", aplanatic, APLANAT_APERTURE),
    )
    for name, field, aperture in rows:
        real = np.abs(field.electric.values).max(axis=(0, 1))
        ideal = np.abs(ideal_focus(field.grid, aperture).electric.values)
        ideal = ideal.max(axis=(0, 1))
        print(
            f"{name}: max |Ex|, |Ey|, |Ez| = {real[0]:.1f}, {real[1]:.2e}, "
            f"{real[2]:.1f} V/m; its ideal lens: {ideal[0]:.1f}, {ideal[1]:.2e}, "
            f"{ideal[2]:.1f} V/m"
        )
    check = debye_aplanat()
    print(
        "aplanatic lens, ray-optics Debye sum: "
        f"{check[0]:.1f}, {check[1]:.2e}, {check[2]:.1f} V/m"
    )
    if figures is None:
        return
    figures = Path(figures)
    wanted = set(only) or {"meridional", "refraction", "focal", "lenses", "mie"}
    if "focal" in wanted:
        figure_focal(fields, figures / "vector_focal_polarization.png")
    if "lenses" in wanted:
        figure_lenses(
            fields["linear x"], aplanatic, figures / "vector_real_vs_ideal.png"
        )
    if "refraction" in wanted:
        figure_refraction(carried["linear x"], figures / "vector_refraction.png")
    if "mie" in wanted:
        errors = figure_mie(figures / "vector_mie_validation.png")
        for radius, error in errors.items():
            print(f"aplanatic sphere R = {radius:g} wavelengths: error {error:.2e}")
    if "meridional" in wanted:
        figure_meridional(carried, figures / "vector_meridional.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
