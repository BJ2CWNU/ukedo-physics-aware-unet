"""
Fig 4 plot — Qualitative reconstruction example for split=10, model_seed=42.

Caption (matches manuscript verbatim):
    Upper row (surface domain):
        (a) sparse aerial input (n = 1,106, circles)
        (b) IDW surface proxy
        (c) AI ground estimate
    Lower row (aerial domain):
        (d) held-out observed (n = 1,107, squares)
        (e) IDW aerial prediction
        (f) AI aerial prediction

Required input arrays (produced by predict_split10_seed42.py):
    fig45_data/split10_seed42_sparse_input.npy
    fig45_data/split10_seed42_idw_surface.npy
    fig45_data/split10_seed42_unet_surface.npy
    fig45_data/split10_seed42_idw_aerial.npy
    fig45_data/split10_seed42_unet_aerial.npy
    fig45_data/split10_seed42_held_out.npz
    fig45_data/split10_seed42_input_points.npz
    fig45_data/split10_seed42_meta.json

Output:
    data/results_v81/Figure_4.png

Usage on workstation:
    cd radiation-mapping/ukedo_river2
    python tools/make_figures/plot_fig4.py
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # ukedo_river2/
DATA_DIR = os.path.join(ROOT, 'data', 'results_v81', 'fig45_data')
OUT = os.path.join(ROOT, 'data', 'results_v81', 'Figure_4.png')

# Caption-derived colorbar ranges (manuscript Fig 4 caption):
#   "The surface-domain colorbar (1,809–25,000 CPS) substantially exceeds
#    the aerial-domain range (1,809–11,200 CPS)"
SURF_VMIN, SURF_VMAX = 1809, 25000
AERIAL_VMIN, AERIAL_VMAX = 1809, 11200

# Mask threshold for "outside UAV trajectory coverage" — from caption:
#   "Grey shading denotes region outside UAV trajectory coverage"
# We approximate the coverage region as cells where the IDW interpolation has
# weight (i.e., at least one nearby measurement). For practical implementation,
# any cell within a few cells of the sparse measurement points is considered
# covered. The simplest robust definition: cells where IDW != 0.
COVERAGE_KERNEL_RADIUS_CELLS = 3  # ~30 m at 10 m/cell


def load_arrays():
    files = {
        'sparse':         'split10_seed42_sparse_input.npy',
        'idw_surface':    'split10_seed42_idw_surface.npy',
        'unet_surface':   'split10_seed42_unet_surface.npy',
        'idw_aerial':     'split10_seed42_idw_aerial.npy',
        'unet_aerial':    'split10_seed42_unet_aerial.npy',
    }
    arrs = {}
    for k, fname in files.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}\nRun predict_split10_seed42.py first.")
        arrs[k] = np.load(path)
    held_out = np.load(os.path.join(DATA_DIR, 'split10_seed42_held_out.npz'))
    input_pts = np.load(os.path.join(DATA_DIR, 'split10_seed42_input_points.npz'))
    with open(os.path.join(DATA_DIR, 'split10_seed42_meta.json')) as f:
        meta = json.load(f)
    return arrs, held_out, input_pts, meta


def coverage_mask(sparse_grid, radius=COVERAGE_KERNEL_RADIUS_CELLS):
    """Boolean mask of grid cells within `radius` cells of any non-zero sparse cell.
    Cells outside this region are 'extrapolation domain' and shaded grey."""
    H, W = sparse_grid.shape
    has_meas = (sparse_grid > 0)
    # Dilate by radius using a simple distance-transform-style approach.
    # For each cell, check if any cell within radius is True.
    # Implemented via stacked rolls (avoids scipy dependency).
    mask = has_meas.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx > radius * radius:
                continue
            shifted = np.zeros_like(has_meas)
            ys = slice(max(0, dy), H + min(0, dy))
            xs = slice(max(0, dx), W + min(0, dx))
            ys_src = slice(max(0, -dy), H - max(0, dy))
            xs_src = slice(max(0, -dx), W - max(0, dx))
            shifted[ys, xs] = has_meas[ys_src, xs_src]
            mask |= shifted
    return mask


def shade_extrapolation(ax, grid, cov_mask, extent):
    """Overlay grey hatching on cells outside UAV coverage."""
    grey_layer = np.where(cov_mask, np.nan, 1.0)
    ax.imshow(grey_layer,
              extent=extent, origin='lower',
              cmap='Greys', vmin=0, vmax=2.5, alpha=0.55,
              aspect='auto', interpolation='none', zorder=2)


def plot_panel(ax, grid, cov_mask, extent, vmin, vmax, title,
               cmap='inferno', cbar_label=None):
    """Draw a 64x128 grid panel with extrapolation shading."""
    masked_grid = np.where(cov_mask, grid, np.nan)
    im = ax.imshow(masked_grid,
                   extent=extent, origin='lower',
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect='auto', interpolation='nearest', zorder=1)
    shade_extrapolation(ax, grid, cov_mask, extent)
    ax.set_title(title, loc='left', fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=6.5)
    return im


def plot_panel_scatter(ax, lat, lon, vals, marker, extent, vmin, vmax,
                       cov_mask, title, cmap='inferno'):
    """Draw a scatter overlay (sparse input or held-out observations)."""
    # Background: extrapolation shading + faint coverage tint
    H, W = cov_mask.shape
    cov_layer = cov_mask.astype(float)
    cov_layer[~cov_mask] = np.nan
    ax.imshow(cov_layer,
              extent=extent, origin='lower',
              cmap='Greys', vmin=0, vmax=8.0, alpha=0.15,
              aspect='auto', interpolation='nearest', zorder=1)
    shade_extrapolation(ax, np.zeros((H, W)), cov_mask, extent)
    sc = ax.scatter(lon, lat, c=vals, marker=marker, s=4.0,
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolors='none', alpha=0.95, zorder=5)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title(title, loc='left', fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=6.5)
    return sc


def main():
    print(f"Plotting Fig 4 from {DATA_DIR}/")
    arrs, held_out, input_pts, meta = load_arrays()

    # Coverage mask derived from sparse input
    cov = coverage_mask(arrs['sparse'])

    extent = [meta['lon_min'], meta['lon_max'],
              meta['lat_min'], meta['lat_max']]

    # ----- Figure layout: 2 rows x 3 cols + colourbar columns -----
    fig = plt.figure(figsize=(7.5, 4.0), dpi=300)
    gs = GridSpec(2, 4, width_ratios=[1, 1, 1, 0.04],
                  hspace=0.28, wspace=0.20)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5})

    # ===== Upper row: surface domain =====
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    cax_top = fig.add_subplot(gs[0, 3])

    # (a) Sparse aerial input — scatter (circles) of sparse measurement points
    sc_a = plot_panel_scatter(
        ax_a, input_pts['lat'], input_pts['lon'], input_pts['cps'],
        marker='o', extent=extent,
        vmin=SURF_VMIN, vmax=SURF_VMAX, cov_mask=cov,
        title=f'(a) Sparse aerial (n={int(input_pts["n_input"])})',
    )

    # (b) IDW surface proxy
    plot_panel(ax_b, arrs['idw_surface'], cov, extent,
               SURF_VMIN, SURF_VMAX,
               title='(b) IDW surface proxy')

    # (c) AI ground estimate
    im_c = plot_panel(ax_c, arrs['unet_surface'], cov, extent,
                      SURF_VMIN, SURF_VMAX,
                      title='(c) AI ground estimate')

    cbar_top = fig.colorbar(im_c, cax=cax_top)
    cbar_top.set_label('Surface CPS', fontsize=7.5)
    cbar_top.ax.tick_params(labelsize=6.5)

    # ===== Lower row: aerial domain =====
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
    cax_bot = fig.add_subplot(gs[1, 3])

    # (d) Held-out observed (scatter, squares)
    plot_panel_scatter(
        ax_d, held_out['lat'], held_out['lon'], held_out['observed'],
        marker='s', extent=extent,
        vmin=AERIAL_VMIN, vmax=AERIAL_VMAX, cov_mask=cov,
        title=f'(d) Held-out (n={int(held_out["n_held_out"])})',
    )

    # (e) IDW aerial prediction
    plot_panel(ax_e, arrs['idw_aerial'], cov, extent,
               AERIAL_VMIN, AERIAL_VMAX,
               title='(e) IDW aerial prediction')

    # (f) AI aerial prediction
    im_f = plot_panel(ax_f, arrs['unet_aerial'], cov, extent,
                      AERIAL_VMIN, AERIAL_VMAX,
                      title='(f) AI aerial prediction')

    cbar_bot = fig.colorbar(im_f, cax=cax_bot)
    cbar_bot.set_label('Aerial CPS', fontsize=7.5)
    cbar_bot.ax.tick_params(labelsize=6.5)

    # Axis labels: only on outermost panels
    for ax in (ax_d, ax_e, ax_f):
        ax.set_xlabel('Longitude (°E)', fontsize=7)
    for ax in (ax_a, ax_d):
        ax.set_ylabel('Latitude (°N)', fontsize=7)

    # Hide tick labels on inner panels for compactness
    for ax in (ax_b, ax_c, ax_e, ax_f):
        ax.tick_params(labelleft=False)
    for ax in (ax_a, ax_b, ax_c):
        ax.tick_params(labelbottom=False)

    # Disable matplotlib offset notation (+3.748e1 / +1.41e2) — show full lat/lon.
    # Limit tick count + rotate so the full-precision labels stay legible.
    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f):
        ax.ticklabel_format(useOffset=False, style='plain', axis='both')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')

    plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Saved: {OUT}")


if __name__ == '__main__':
    main()
