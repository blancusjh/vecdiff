from .CartesianSurfaces import CartesianSurface
from .fields import Field, FieldCircular, FieldCartesian, FieldPolar
from .fresnel import FresnelOvoid, FresnelOvoidParax
from .fourier import (
    FFT2,
    FT,
    FT2,
    FT3,
    IFFT2,
    IFT,
    IFT2,
    IFT3,
    KAXIS,
    KGRID2,
    KGRID3,
    frequency_grid,
)
from .geometry import (
    RayGeometry,
    ReferenceSphere,
    TangentPlane,
    grazing_radius,
    image_sphere,
    incident_sphere,
    ray_geometry,
    ray_geometry_from_rho,
)
from .grid import Grid
from .hankel import HankelTransform, HT_N
from .system import (
    ApertureBudget,
    element_aperture,
    relay_throughput,
    surface_intersection_radius,
)
from .transfer import (
    InterfaceOperator,
    channel_transmittance,
    focal_channel_weights,
    interface_operator,
    paraxial_channel_weights,
    sphere_transfer_eigenvalues,
)
from .longitudinal import (
    generate_Ez_cartesian,
    generate_Ez_field,
    kz_angular_spectrum,
    spacing_from_cartesian_grid,
)
from .propagation import (
    fresnel_coefficients_on_grid,
    incident_field_on_sphere,
    propagate_to_focal_plane_through_diopter,
    propagate_to_focal_plane_through_diopter_fft,
    transfer_weights_on_grid,
    transverse_diopter_operator,
)
from .pupil_mapping import (
    debye_prefactor,
    pupil_coordinate,
    pupil_extent,
    pupil_transform,
    pupil_weight,
    radius_from_pupil_coordinate,
)
from .coordinate_transformation import polar_grid_to_cartesian_grid
from .polarization import (
    PolarizationData,
    ellipse_parameters,
    polarization_from_components,
    polarization_from_field,
    polarization_map_from_field,
    stokes_parameters,
)
from .animation import (
    animate_harmonic_field,
    harmonic_field_frames,
    save_animation,
)

# The general spectral interface-operator engine (any smooth surface, operator
# composition, projection imaging).  Imported last: its stigmatic bridge leans
# on the exact host modules above, which referee it on the Cartesian oval.
from . import wave
