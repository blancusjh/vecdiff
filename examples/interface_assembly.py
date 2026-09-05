"""A constructed spectral cavity: planar validation and a curved-aperture study."""
import numpy as np
import matplotlib.pyplot as plt
from vecdiff import (Medium,Frame,Plane,SphericalCap,DielectricInterface,InterfaceAssembly,
                    CartesianGrid,plane_wave,sample_surface,propagate_interfaces,
                    LayerStack,propagate_layers)
from ._report import main,relative
from vecdiff.observables.electromagnetism import assembly_boundary_residuals


def run():
    air,glass=Medium(),Medium(4.)
    grid=CartesianGrid.from_spacing(1.2,4)
    planes=(Plane(),Plane(Frame(origin=[0,0,1.])))
    caps=(SphericalCap(80.),SphericalCap(-80.,Frame(origin=[0,0,1.])))
    def assembly(surfaces):
        return InterfaceAssembly((DielectricInterface(surfaces[0],air,glass),DielectricInterface(surfaces[1],glass,air)))
    quadrature=tuple(sample_surface(s,(0,2.),(0,2*np.pi),12,24) for s in caps)
    wavelengths=np.linspace(.94,1.06,13)
    plane_curve=[]; reference_curve=[]; curved_curve=[]; residual=[]; errors=[]
    for wavelength in wavelengths:
        incident=plane_wave(wavelength=wavelength)
        planar=propagate_interfaces(incident,assembly(planes),grid,rtol=1e-10)
        reference=propagate_layers(incident,LayerStack((air,glass,air),(1.,)))
        actual=planar.evaluate([[0,0,2]],region=2)[0]
        exact=reference.evaluate([[0,0,2]],region=2)[0]
        errors.append(relative(actual,exact));plane_curve.append(float(np.sum(abs(actual)**2)))
        reference_curve.append(float(np.sum(abs(exact)**2)))
        curved=propagate_interfaces(incident,assembly(caps),grid,samplings=quadrature,rtol=1e-9)
        curved_curve.append(float(np.sum(abs(curved.evaluate([[0,0,2]],region=2)[0])**2)))
        residual.append(curved.feedback.relative_residual)
    # Refine quadrature without changing either physical aperture or spectral lattice.
    refined=tuple(sample_surface(s,(0,2.),(0,2*np.pi),24,48) for s in caps)
    f=propagate_interfaces(plane_wave(),assembly(caps),grid,samplings=quadrature,rtol=1e-9)
    g=propagate_interfaces(plane_wave(),assembly(caps),grid,samplings=refined,rtol=1e-9)
    change=relative(f.evaluate([[0,0,2]],region=2)[0],g.evaluate([[0,0,2]],region=2)[0])
    assert max(errors)<1e-8 and max(residual)<1e-9 and change<1e-7
    held_out=tuple(sample_surface(s,(.05,1.),(0,2*np.pi),8,16) for s in caps)
    jumps=assembly_boundary_residuals(g,held_out)
    fig,axes=plt.subplots(1,3,figsize=(14,3.7),constrained_layout=True)
    axes[0].plot(wavelengths,reference_curve,label='Independent planar layer recursion')
    axes[0].plot(wavelengths,plane_curve,'o',mfc='none',label='Constructed spectral encounters')
    axes[0].set(xlabel='Vacuum wavelength / reference wavelength',ylabel='On-axis |E|²',title='Planar cavity: independent validation')
    axes[0].legend(fontsize=8)
    axes[1].plot(wavelengths,curved_curve,'o-')
    axes[1].set(xlabel='Vacuum wavelength / reference wavelength',ylabel='On-axis |E|²',title='Curved finite apertures: model study')
    x=np.arange(4)
    for j,row in enumerate(jumps):
        axes[2].bar(x+(j-.5)*.35,list(row.values()),.35,label=f'Interface {j+1}')
    axes[2].set(xticks=x,xticklabels=['Et','Ht','Dn','Bn'],ylabel='Normalized reconstructed jump',title='Curved boundary acceptance is not met')
    axes[2].legend(fontsize=8)
    return fig,dict(parameters=dict(indices=[1,4,1],gap=1.,curvature_radii=[80.,-80.],aperture_radius=2.,
        wavelengths=wavelengths.tolist(),grid_count=4,grid_spacing=1.2),
        assumptions='Curved curves use local physical optics and a finite periodic propagating spectrum. Source quadrature and feedback checks do not certify spectral-window convergence, Maxwell boundary accuracy, or resolved high-Q peaks.',
        max_planar_complex_field_error=max(errors),max_curved_feedback_residual=max(residual),
        curved_surface_quadrature_change=change,reconstructed_boundary=jumps,planar_intensity=plane_curve,curved_intensity=curved_curve)


if __name__=='__main__': main('interface_assembly',run)
