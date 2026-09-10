#!/usr/bin/env python3
"""
lqt_pykonal_tomography_vs_model_depth_slices.py

Plot the Baillard 3D Vs model itself in the SAME visual format as the raylength
anisotropy depth-slice maps, for direct side-by-side comparison.

Unlike the anisotropy maps there is NO period dimension -- the Vs model is
time-invariant -- so this is a single page of 4 depth rows x 1 column. The
per-voxel Vs sampling convention, voxel grid, depth rows, bathymetry, station
and fault overlays are all reused verbatim from
lqt_pykonal_tomography_raylength_anisotropy (imported as `ral`) so the two
figures are guaranteed to be sampled and drawn identically. In particular
ral.vs_flat is Vs sampled at the TRUE voxel centres (xn+VXY/2, yn+VXY/2,
zn+VZ/2) via vs_at(), clipped [0.3, 5.0] -- exactly what the raylength script
uses for A_ray = dt*Vs_avg*100/r.

No coverage gate is applied: the Vs model is defined at every voxel, so every
voxel is shown.

Output:
    lqt_pykonal_combined_results/lqt_pykonal_tomography_vs_model_depth_slices.pdf
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lqt_pykonal_tomography_raylength_anisotropy as ral  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  (ral already set the Agg backend on import)
import matplotlib.lines as mlines  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

OUT_PDF = os.path.join(
    ral.OUT_DIR, 'lqt_pykonal_tomography_vs_model_depth_slices.pdf')

# Perceptually-uniform sequential colormap, DISTINCT from the anisotropy maps'
# Blues so the two figures are never confused at a glance.
CMAP = 'viridis'


def main():
    # Populate ral.vs_flat plus bathymetry / station / fault plotting state.
    # (This also loads the MLdd catalogs and builds the ray tracer, which we
    # don't need here, but it is the single supported entry point for the
    # shared per-voxel Vs sampling and is not expensive relative to plotting.)
    ral._setup_environment()

    # Vs per voxel, flat-index order over (NX, NY, NZ) -> reshape to grid.
    vs_grid = ral.vs_flat.reshape(ral.NX, ral.NY, ral.NZ)

    # Global finite Vs range across ALL four depth rows -> shared colorbar.
    row_slices = [np.nanmean(vs_grid[:, :, iz0:iz1], axis=2)
                  for (iz0, iz1), _zlbl, _z0 in ral.DEPTH_ROWS]
    finite = np.concatenate([s[np.isfinite(s)].ravel() for s in row_slices])
    vmin, vmax = float(finite.min()), float(finite.max())
    levels = np.linspace(vmin, vmax, 15)

    n_dep = len(ral.DEPTH_ROWS)
    fig = plt.figure(figsize=(2.2 + 0.6, n_dep * 3.0 + 0.5))
    gs = GridSpec(n_dep, 2, width_ratios=[1, 0.04], hspace=0.04, wspace=0.04)

    for ri, (((iz0, iz1), zlbl, z0), sl) in enumerate(zip(ral.DEPTH_ROWS, row_slices)):
        ax = fig.add_subplot(gs[ri, 0])
        ral._bathy(ax)
        # Vs is finite everywhere in-grid; plot it directly (no smoothing/mask
        # -- the anisotropy map's smoothing exists to tame sparse ray coverage,
        # which does not apply to a fully-defined model field).
        ax.contourf(ral.Xg, ral.Yg, sl, levels=levels, cmap=CMAP,
                    vmin=vmin, vmax=vmax, extend='neither')
        ral._sta(ax)
        ral._kid(ax)
        ral._faults(ax, z0)
        ax.set_xlim(ral.X_START, ral.X_END)
        ax.set_ylim(ral.Y_START, ral.Y_END)
        ax.set_aspect('equal', 'box')
        ax.tick_params(labelsize=4)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylabel(f'{zlbl}', fontsize=7)
        if ri == 0:
            ax.set_title('Baillard Vs model', fontsize=7.5, fontweight='bold', pad=1)

    sm = ScalarMappable(cmap=plt.colormaps[CMAP], norm=Normalize(vmin, vmax))
    sm.set_array([])
    cax = fig.add_subplot(gs[:, 1])
    cb = plt.colorbar(sm, cax=cax, ticks=np.linspace(vmin, vmax, 5),
                      format=FormatStrFormatter('%.2f'))
    cb.set_label('Baillard Vs model, mean per voxel [km/s]', fontsize=9)
    cb.ax.tick_params(labelsize=7)

    legend_handles = [
        mlines.Line2D([], [], marker='o', color='w', mfc='red', mec='k', ms=5, label='Kidiwela'),
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6, label='OBS station'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults (depth-corrected)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.45, 0.0))
    fig.suptitle('Baillard 3D Vs model, sampled at voxel centres (0.25 km grid)\n'
                 'for comparison against the LQT + PyKonal-FMM anisotropy maps',
                 fontsize=10, fontweight='bold', y=0.99)
    fig.subplots_adjust(bottom=0.06)

    print(f'Writing {os.path.basename(OUT_PDF)}...')
    fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
