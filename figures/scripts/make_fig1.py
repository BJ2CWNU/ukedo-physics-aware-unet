"""
Fig 1 generator — Study area and flight geometry of the Fukushima Ukedo benchmark.

Caption:
    (a) Location relative to FDNPS.
    (b) UAV flight trajectory with CPS overlay, shown on the 64×128 analysis
        grid with the land–water boundary used by the physics-aware model.
        Flight altitude 30 m AGL, line spacing 30 m, speed ~5 km/h
        (n = 2,213 points).

Required files:
    - data/integrated data.csv          (lat, lon, time, value)
    - data/ukedo_water_mask.png         (binary water mask 64×128)

Output:
    - results_v81/Figure_1.png          (300 DPI, ~17 cm wide, 2-panel)

Usage on workstation:
    cd radiation-mapping/ukedo_river2
    python tools/make_figures/make_fig1.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from PIL import Image

# ============================================================================
# Configuration
# ============================================================================
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_CSV = os.path.join(ROOT, 'data', 'integrated data.csv')
WATER_MASK_PNG = os.path.join(ROOT, 'data', 'ukedo_water_mask.png')
OUT_DIR = os.path.join(ROOT, 'data', 'results_v81')
OUT = os.path.join(OUT_DIR, 'Figure_1.png')

# Fukushima Daiichi Nuclear Power Station coordinates (publicly known)
FDNPS_LAT = 37.4226
FDNPS_LON = 141.0331

# CSV column mapping (matches v8 config)
CSV_TIME_COL  = 'time'
CSV_LAT_COL   = 'lat'
CSV_LON_COL   = 'lon'
CSV_VALUE_COL = 'value'


# ============================================================================
# Data loading
# ============================================================================
def load_trajectory():
    """Load Ukedo trajectory CSV. Returns DataFrame with lat, lon, value, time."""
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(
            f"Trajectory CSV not found at {DATA_CSV}.\n"
            f"Place the integrated data CSV in data/integrated data.csv."
        )
    df = pd.read_csv(DATA_CSV)
    # Detect column names if non-standard (some versions use 'cps' instead of 'value')
    col_map = {}
    for col_choice, candidates in {
        'lat':   ['lat', 'latitude', 'Latitude'],
        'lon':   ['lon', 'longitude', 'Longitude'],
        'value': ['value', 'cps', 'CPS', 'count_rate'],
        'time':  ['time', 'Time', 'datetime', 'timestamp'],
    }.items():
        for cand in candidates:
            if cand in df.columns:
                col_map[col_choice] = cand
                break
    if 'lat' not in col_map or 'lon' not in col_map or 'value' not in col_map:
        raise ValueError(f"Cannot identify lat/lon/value columns. CSV columns: {df.columns.tolist()}")
    df = df.rename(columns={col_map['lat']: 'lat', col_map['lon']: 'lon',
                            col_map['value']: 'value'})
    return df


def load_water_mask():
    """Load binary land-water mask. Returns 2D array (64, 128) where 1=water."""
    if not os.path.exists(WATER_MASK_PNG):
        print(f"  Warning: water mask not found at {WATER_MASK_PNG}; skipping overlay.")
        return None
    img = np.array(Image.open(WATER_MASK_PNG).convert('L'))
    # Binary: water pixels are dark in the mask file
    # Threshold at 128
    mask = (img < 128).astype(np.uint8)
    return mask


# ============================================================================
# Plotting
# ============================================================================
def make_figure(df, water_mask):
    n_points = len(df)
    print(f"  Trajectory points: {n_points}")
    print(f"  CPS range: {df['value'].min():.1f} – {df['value'].max():.1f}")
    print(f"  Lat range: {df['lat'].min():.4f} – {df['lat'].max():.4f}")
    print(f"  Lon range: {df['lon'].min():.4f} – {df['lon'].max():.4f}")
    
    fig = plt.figure(figsize=(7.0, 3.5), dpi=300)
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.4], wspace=0.25)
    
    # ---------------------------------------------------------------------
    # Panel (a): Location relative to FDNPS
    # ---------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    
    # Compute trajectory bounding box and centre
    survey_lat_centre = df['lat'].mean()
    survey_lon_centre = df['lon'].mean()
    
    # Plot extent: span from FDNPS to survey area with margin
    lat_min = min(FDNPS_LAT, df['lat'].min()) - 0.02
    lat_max = max(FDNPS_LAT, df['lat'].max()) + 0.02
    lon_min = min(FDNPS_LON, df['lon'].min()) - 0.02
    lon_max = max(FDNPS_LON, df['lon'].max()) + 0.02
    
    # Light grey background for land context
    ax1.set_facecolor('#FAFAFA')
    
    # FDNPS marker
    ax1.scatter([FDNPS_LON], [FDNPS_LAT], marker='*', s=200, c='red',
                edgecolors='black', linewidths=0.8, zorder=10, label='FDNPS')
    ax1.annotate('FDNPS', (FDNPS_LON, FDNPS_LAT),
                 xytext=(8, -10), textcoords='offset points',
                 fontsize=9, fontweight='bold', color='red')
    
    # Survey area as a rectangle
    survey_box = mpatches.Rectangle(
        (df['lon'].min(), df['lat'].min()),
        df['lon'].max() - df['lon'].min(),
        df['lat'].max() - df['lat'].min(),
        linewidth=1.5, edgecolor='#1565C0', facecolor='#BBDEFB', alpha=0.6,
        zorder=5,
    )
    ax1.add_patch(survey_box)
    ax1.annotate('Survey area\n(Ukedo basin)',
                 (survey_lon_centre, df['lat'].max()),
                 xytext=(0, 10), textcoords='offset points',
                 fontsize=8, ha='center', color='#1565C0', fontweight='bold')
    
    # Distance line
    ax1.plot([FDNPS_LON, survey_lon_centre], [FDNPS_LAT, survey_lat_centre],
             linestyle='--', color='grey', linewidth=0.8, zorder=4)
    # Compute approximate distance (rough — assumes flat earth at this latitude)
    dlat = (survey_lat_centre - FDNPS_LAT)
    dlon = (survey_lon_centre - FDNPS_LON)
    dist_km = np.sqrt((dlat * 111.32) ** 2
                       + (dlon * 111.32 * np.cos(np.radians(FDNPS_LAT))) ** 2)
    midlat = (FDNPS_LAT + survey_lat_centre) / 2
    midlon = (FDNPS_LON + survey_lon_centre) / 2
    ax1.text(midlon + 0.005, midlat,
             f'~{dist_km:.0f} km',
             fontsize=7, color='grey', style='italic')
    
    ax1.set_xlim(lon_min, lon_max)
    ax1.set_ylim(lat_min, lat_max)
    ax1.set_xlabel('Longitude (°E)', fontsize=8)
    ax1.set_ylabel('Latitude (°N)', fontsize=8)
    ax1.tick_params(labelsize=7)
    # Disable matplotlib's auto-offset so axes show full lat/lon (e.g., 37.48
    # rather than '+3.748e1' placed in a corner).
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title('(a)', loc='left', fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle=':', linewidth=0.4, alpha=0.6)
    
    # ---------------------------------------------------------------------
    # Panel (b): UAV flight trajectory with CPS overlay on 64×128 grid
    # ---------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    
    # Land-water mask: display water as light grey hatched region (not blue —
    # avoids colour clash with both the panel-(a) survey-area box and the
    # trajectory CPS colormap).
    if water_mask is not None:
        H, W = water_mask.shape
        extent = [df['lon'].min(), df['lon'].max(),
                  df['lat'].min(), df['lat'].max()]
        # Render water cells as a faint grey wash; non-water cells transparent
        water_layer = np.where(water_mask > 0.5, 1.0, np.nan)
        ax2.imshow(water_layer, extent=extent, origin='lower',
                   cmap='Greys', vmin=0, vmax=2.5, alpha=0.40,
                   aspect='auto', zorder=2)
        # Add a hatched overlay for clearer water identification
        # (use a contour-like approach: imshow with hatch via PatchCollection
        # is non-trivial; we use an additional darker imshow band)
        water_outline = np.where(water_mask > 0.5, 1.0, np.nan)
        ax2.imshow(water_outline, extent=extent, origin='lower',
                   cmap='Greys', vmin=-0.8, vmax=2.5, alpha=0.18,
                   aspect='auto', zorder=2.5)
    
    # Trajectory points coloured by CPS — slightly larger markers + edge for
    # visibility against the grey water background.
    sc = ax2.scatter(df['lon'], df['lat'], c=df['value'],
                     s=4.5, cmap='inferno', alpha=0.95,
                     edgecolors='none', zorder=5,
                     vmin=df['value'].min(),
                     vmax=df['value'].quantile(0.99))
    
    cbar = plt.colorbar(sc, ax=ax2, fraction=0.04, pad=0.02)
    cbar.set_label('CPS', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    
    ax2.set_xlim(df['lon'].min(), df['lon'].max())
    ax2.set_ylim(df['lat'].min(), df['lat'].max())
    ax2.set_xlabel('Longitude (°E)', fontsize=8)
    ax2.set_ylabel('Latitude (°N)', fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.ticklabel_format(useOffset=False, style='plain')
    ax2.set_title('(b)', loc='left', fontsize=10, fontweight='bold')
    
    # Annotation: flight parameters — placed at lower-right (away from dense
    # northern trajectory points). Strong opaque white background to ensure
    # readability over the water mask shading.
    ax2.text(0.97, 0.03,
             f'Altitude: 30 m AGL\nLine spacing: 30 m\nSpeed: ~5 km/h\nn = {n_points:,} points',
             transform=ax2.transAxes, fontsize=7, va='bottom', ha='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor='#666', linewidth=0.6, alpha=0.96),
             zorder=20)
    
    plt.tight_layout(pad=0.4)
    plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n  Saved: {OUT}")


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("Generating Figure 1 — Study area and flight geometry")
    print(f"  ROOT: {ROOT}")
    print(f"  CSV:  {DATA_CSV}")
    print()
    
    df = load_trajectory()
    water_mask = load_water_mask()
    
    os.makedirs(OUT_DIR, exist_ok=True)
    make_figure(df, water_mask)
