"""
Fig 5 plot — Scatter plots of predicted vs observed aerial CPS.
split=10, model_seed=42 (single model).

Three panels:
    (a) IDW prediction vs observed (n = 1,107)
    (b) Kriging prediction vs observed
    (c) U-Net (single model, model_seed=42) prediction vs observed

Required input (produced by predict_split10_seed42.py):
    fig45_data/split10_seed42_held_out.npz
        keys: observed, idw_pred, krg_pred, ai_pred, lat, lon, n_held_out
    fig45_data/split10_seed42_meta.json    (for RMSE annotations)

Output:
    data/results_v81/Figure_5.png
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA_DIR = os.path.join(ROOT, 'data', 'results_v81', 'fig45_data')
OUT = os.path.join(ROOT, 'data', 'results_v81', 'Figure_5.png')

# Caption-derived axis ranges:
#   IDW predictions saturate near 5800 CPS; observed up to 11,184.
#   Use full observed range (1809 - ~11500) symmetric for both axes
#   so the 1:1 line spans diagonally.
AXIS_MIN = 1500
AXIS_MAX = 11500


def compute_metrics(observed, predicted):
    """Compute RMSE, Pearson r, MBE."""
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    diff = pred - obs
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mbe = float(np.mean(diff))
    if np.std(obs) > 0 and np.std(pred) > 0:
        r = float(np.corrcoef(obs, pred)[0, 1])
    else:
        r = float('nan')
    return rmse, r, mbe


def plot_scatter(ax, observed, predicted, color, title, label_metrics=True,
                 method_name=''):
    """Single scatter panel with 1:1 dashed line + metrics annotation."""
    rmse, r, mbe = compute_metrics(observed, predicted)
    print(f"    {method_name:>20s}: RMSE={rmse:.1f}, r={r:.3f}, MBE={mbe:+.1f}")

    # Density-aware alpha to avoid overplotting
    n = len(observed)
    alpha = max(0.18, min(0.85, 600 / n))

    ax.scatter(observed, predicted,
               s=5.0, c=color, alpha=alpha, edgecolors='none')

    # 1:1 dashed reference
    ax.plot([AXIS_MIN, AXIS_MAX], [AXIS_MIN, AXIS_MAX],
            linestyle='--', color='black', linewidth=0.7, alpha=0.7,
            label='y = x', zorder=10)

    ax.set_xlim(AXIS_MIN, AXIS_MAX)
    ax.set_ylim(AXIS_MIN, AXIS_MAX)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, loc='left', fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.5)

    if label_metrics:
        text = (f'$n$ = {n:,}\n'
                f'RMSE = {rmse:.0f} CPS\n'
                f'$r$ = {r:.2f}\n'
                f'MBE = {mbe:+.0f}')
        ax.text(0.04, 0.96, text,
                transform=ax.transAxes, fontsize=6.8,
                ha='left', va='top', family='DejaVu Sans',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='white', edgecolor='#999',
                          linewidth=0.5, alpha=0.92))


def main():
    print(f"Plotting Fig 5 from {DATA_DIR}/")

    held_out_path = os.path.join(DATA_DIR, 'split10_seed42_held_out.npz')
    if not os.path.exists(held_out_path):
        raise FileNotFoundError(
            f"Missing {held_out_path}\nRun predict_split10_seed42.py first.")
    data = np.load(held_out_path)

    observed = data['observed']
    idw_pred = data['idw_pred']
    krg_pred = data['krg_pred']
    ai_pred = data['ai_pred']
    n = len(observed)
    print(f"  Held-out points: n = {n}")
    print(f"  Computing metrics:")

    # ---------------- Figure layout ----------------
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.0})
    fig = plt.figure(figsize=(7.5, 2.7), dpi=300)
    gs = GridSpec(1, 3, wspace=0.32)

    ax_a = fig.add_subplot(gs[0])
    plot_scatter(ax_a, observed, idw_pred,
                 color='#D32F2F',
                 title='(a) IDW',
                 method_name='IDW')

    ax_b = fig.add_subplot(gs[1])
    plot_scatter(ax_b, observed, krg_pred,
                 color='#388E3C',
                 title='(b) Ordinary kriging',
                 method_name='Kriging')

    ax_c = fig.add_subplot(gs[2])
    plot_scatter(ax_c, observed, ai_pred,
                 color='#1565C0',
                 title='(c) Physics-aware U-Net (seed=42)',
                 method_name='U-Net')

    # Shared axis labels (LaTeX italicized variables)
    for ax in (ax_a, ax_b, ax_c):
        ax.set_xlabel('Observed aerial CPS', fontsize=7.5)
    ax_a.set_ylabel('Predicted aerial CPS', fontsize=7.5)

    plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Saved: {OUT}")


if __name__ == '__main__':
    main()
