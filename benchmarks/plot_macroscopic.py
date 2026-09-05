"""Plot recorded spectral-method measurements; never recompute or relabel them."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def run():
    path=Path(__file__).parent/'results/macroscopic.json'
    report=json.loads(path.read_text())
    rows=[r for r in report['cases'] if r['refinement']=='combined']
    radii=[r['radius_over_wavelength'] for r in rows]
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),constrained_layout=True)
    for key,label in [('tangential_E','Et'),('tangential_H','Ht'),('normal_D','Dn'),('normal_B','Bn')]:
        axes[0].plot(radii,[100*r['reconstructed_boundary'][key] for r in rows],'o-',label=label)
    axes[0].set(xlabel='Curvature radius / vacuum wavelength',ylabel='Normalized boundary jump (%)',title='Boundary accuracy remains incomplete')
    axes[0].legend(ncol=2,fontsize=8)
    axes[1].plot(radii,[100*r['power_balance_error'] for r in rows],'o-')
    axes[1].set(xlabel='Curvature radius / vacuum wavelength',ylabel='|R + T − 1| (%)',title='Integrated propagating power')
    for name in ('surface','polar','table','combined'):
        subset=[r for r in report['cases'] if r['refinement']==name]
        axes[2].semilogy([r['radius_over_wavelength'] for r in subset],
                       [max(100*r['downstream_field_change'],1e-12) for r in subset],'o-',label=name)
    axes[2].set(xlabel='Curvature radius / vacuum wavelength',ylabel='Downstream field change (%)',title='Independent numerical controls')
    axes[2].legend(fontsize=8)
    return fig,report


if __name__=='__main__':
    fig,_=run()
    fig.savefig(Path(__file__).resolve().parents[1]/'docs/assets/macroscopic_validation.png',dpi=150)
