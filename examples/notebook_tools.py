"""Plotting conventions shared by the application notebooks; no hidden physics."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from vecdiff.fields.polarization import stokes


def style():
    plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 180, 'font.size': 11,
                         'axes.titlesize': 12, 'axes.labelsize': 11,
                         'image.cmap': 'hot', 'axes.spines.top': False,
                         'axes.spines.right': False})


def show(fig, name):
    """Embed the exact figure and save a reproducible review PNG."""
    from IPython.display import display
    out = Path('build/notebook-review'); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/(name+'.png'), dpi=180, bbox_inches='tight')
    display(fig)
    plt.close(fig)


def scalar_map(fig, ax, values, x, y, title, *, label=r'$|\mathbf{E}|^2/|E_0|^2$',
               xlabel=r'$x$ (µm)', ylabel=r'$z$ (µm)', vmax=None, vmin=0, cmap='hot', norm=None):
    image = ax.pcolormesh(x, y, values, shading='auto', cmap=cmap,
                         **({'vmin':vmin,'vmax':vmax} if norm is None else {'norm':norm}), rasterized=True)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig.colorbar(image, ax=ax, label=label, shrink=.85)
    return image


def polarization_map(fig, ax, e, x, y, *, title='Transverse polarization', threshold=.01):
    s = stokes(e[..., 0], e[..., 1])
    psi = .5*np.arctan2(s[..., 2], s[..., 1])
    chi = .5*np.arcsin(np.clip(s[..., 3]/np.maximum(s[..., 0], 1e-300), -1, 1))
    valid = s[..., 0] > threshold*s[..., 0].max()
    scalar_map(fig, ax, np.where(valid, np.degrees(chi), np.nan), x, y, title,
               label='Ellipticity angle (degrees)', xlabel='x (µm)', ylabel='y (µm)',
               vmin=-45, vmax=45, cmap='coolwarm')
    stride = max(1, len(x)//17); width = (x[-1]-x[0])/27
    for j in range(stride//2, len(y), stride):
        for i in range(stride//2, len(x), stride):
            if valid[j,i]:
                ax.add_patch(Ellipse((x[i], y[j]), width, max(.012, abs(np.tan(chi[j,i])))*width,
                    angle=np.degrees(psi[j,i]), fill=False, edgecolor='black', linewidth=.7))
    ax.set_aspect('equal')


def fwhm(x, values):
    """Interpolated width of the connected half-maximum peak."""
    x, values = np.asarray(x), np.asarray(values)
    i = int(np.argmax(values)); half = values[i]/2
    left = np.flatnonzero(values[:i] < half)
    right = np.flatnonzero(values[i+1:] < half)
    if not len(left) or not len(right):
        raise ValueError('The observation window does not contain both half-maximum crossings')
    j, k = left[-1], i+1+right[0]
    xl = x[j]+(half-values[j])*(x[j+1]-x[j])/(values[j+1]-values[j])
    xr = x[k-1]+(half-values[k-1])*(x[k]-x[k-1])/(values[k]-values[k-1])
    return float(xr-xl)
