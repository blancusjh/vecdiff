"""Complete a sampled electric field, then propagate its Maxwell spectrum."""
import numpy as np
import matplotlib.pyplot as plt
from vecdiff import CartesianGrid, PlaneDomain, TransverseElectricField, propagate, spectrum_of
from vecdiff.observables.electromagnetism import poynting
from ._report import main, image, relative


def run():
    """Periodic Gaussian beam: Ez completion, reversibility, and flux conservation."""
    # This lattice has no evanescent bins, making reverse propagation stable.
    # Finer lattices require an explicit regularization for backward continuation.
    grid = CartesianGrid.from_spacing(.8, 64)
    x, y = grid.xy
    pupil = np.exp(-(x*x+y*y)/3**2)
    field = TransverseElectricField(pupil, np.zeros_like(pupil), grid, PlaneDomain())
    complete = field.complete()
    distance = 20.
    propagated = propagate(complete, distance)
    returned = propagate(propagated, -distance)
    spec = spectrum_of(complete)
    powers = []
    for domain in (complete.domain, propagated.domain):
        e, h = spec.evaluate(domain.points(grid))
        powers.append(float(np.sum(poynting(e, h)[..., 2])*grid.dx*grid.dy))
    reversal = relative(returned.components, complete.components)
    flux_error = abs(powers[1]/powers[0]-1)
    assert reversal < 1e-9 and flux_error < 1e-9
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    for ax, values, title in zip(axes, [complete.norm2(), abs(complete.Ez)**2, propagated.norm2()],
                                  ["Input |E|²", "Completed |Ez|²", "Propagated |E|², z=20 λ"]):
        fig.colorbar(image(ax, grid, values, title), ax=ax, shrink=.8)
        ax.set(xlim=(-8, 8), ylim=(-8, 8))
    return fig, dict(purpose="Unknown Ez is completed through k·E=0, not set to zero.",
                    parameters=dict(wavelength=1., waist=3., spacing=.8, count=64, distance=distance),
                    assumptions="Homogeneous medium; periodic, propagating-only sampled model. Not a hard-aperture model.",
                    reversal_relative_error=reversal, normal_flux_relative_error=flux_error,
                    longitudinal_norm_fraction=float(np.sum(abs(complete.Ez)**2)/np.sum(complete.norm2())))


if __name__ == "__main__":
    main("field_propagation", run)
