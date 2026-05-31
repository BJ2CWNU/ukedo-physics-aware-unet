"""
b1_psf_free.py — Phase B1: PSF-free IDW/Kriging Supplementary

Goal: Address the reviewer question "why apply PSF to IDW twice?" by reporting
both evaluation frames side by side.

Frame A (main, manuscript): IDW/Kriging treated as surface estimates,
  forward-projected through PSF, then compared to held-out aerial CPS.
  This is the "double-blurring" frame (manuscript Sec 1.3).

Frame B (supplementary): IDW/Kriging treated as direct aerial interpolators,
  no PSF applied, compared directly to held-out aerial CPS.
  This is the "aerial interpolation accuracy" task — a different scientific
  question than what the paper studies.

U-Net is NOT evaluated under Frame B because its task is surface estimation;
Frame B asks a question U-Net is not designed to answer. This asymmetry is
intentional and is the supplementary's main pedagogical point.

Usage:
  python b1_psf_free.py                      # uses 5 split seeds, no model needed
  python b1_psf_free.py --splits 10 20 30 40 50

Output:
  results_v81/b1_psf_free.json
  results_v81/fig_b1_two_frames.png   (RMSE bar plot, 2 frames x 3 methods)
  results_v81/fig_b1_scatter.png      (PSF vs no-PSF scatter, 1 split)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Reuse v8.1 primitives
from ukedo_v81 import (
    Config, get_physics_kernel, load_terrain,
    split_real_data, grid_coords, compute_idw_torch, compute_kriging,
    compute_metrics,
)


# =============================================================================
# Two-frame evaluation (no model needed - IDW/Kriging only)
# =============================================================================
def evaluate_two_frames(split_seed, df_full, cfg, kernel, water_mask):
    """Return per-split results for both Frame A (PSF) and Frame B (no PSF).

    Frame A: surface = IDW (or Kriging) -> forward-conv with PSF -> sample at held-out points
    Frame B: aerial  = IDW (or Kriging) directly                  -> sample at held-out points

    Returns dict with:
      true, idw_A, idw_B, krg_A, krg_B  (all numpy arrays of length n_held_out)
    """
    H, W = cfg.H, cfg.W
    device = cfg.DEVICE
    pad_y, pad_x = H - 1, W - 1

    df_in, df_out = split_real_data(df_full, cfg.HOLDOUT_RATIO, seed=split_seed)
    lat_min = df_full['Latitude'].min(); lat_max = df_full['Latitude'].max()
    lon_min = df_full['Longitude'].min(); lon_max = df_full['Longitude'].max()

    y_in, x_in = grid_coords(df_in, lat_min, lat_max, lon_min, lon_max, H, W)
    sparse_map = torch.zeros((H, W), device=device)
    count_map = torch.zeros((H, W), device=device)
    for yi, xi, cps in zip(y_in, x_in, df_in['CPS'].values):
        sparse_map[yi, xi] += float(cps)
        count_map[yi, xi] += 1
    meas_mask = (count_map > 0).float()
    sparse_map[meas_mask > 0] /= count_map[meas_mask > 0]
    MAX_CPS = sparse_map.max().item() if sparse_map.max().item() > 0 else 1.0
    sparse_norm = sparse_map / MAX_CPS

    # IDW & Kriging (in normalized scale, then upscale)
    idw_norm = compute_idw_torch(meas_mask, sparse_norm, cfg)
    krig_map_np = compute_kriging(meas_mask.cpu().numpy(), sparse_map.cpu().numpy(), cfg)
    krig_norm = torch.from_numpy(krig_map_np / MAX_CPS).float().to(device).clamp(min=0.0)

    # Frame A: PSF-applied (treat as surface)
    surf_IDW = idw_norm.view(1, 1, H, W)
    surf_KRG = krig_norm.view(1, 1, H, W)
    aerial_IDW_A = F.conv2d(surf_IDW, kernel, padding=(pad_y, pad_x)).squeeze() * MAX_CPS
    aerial_KRG_A = F.conv2d(surf_KRG, kernel, padding=(pad_y, pad_x)).squeeze() * MAX_CPS

    # Frame B: PSF-free (treat as direct aerial interpolation)
    aerial_IDW_B = idw_norm * MAX_CPS
    aerial_KRG_B = krig_norm * MAX_CPS

    # Sample at held-out trajectory points
    y_out, x_out = grid_coords(df_out, lat_min, lat_max, lon_min, lon_max, H, W)
    true_vals = df_out['CPS'].values.astype(float)

    return {
        'split_seed': int(split_seed),
        'n_held_out': int(len(true_vals)),
        'true': true_vals,
        'idw_A': aerial_IDW_A[y_out, x_out].cpu().numpy(),
        'idw_B': aerial_IDW_B[y_out, x_out].cpu().numpy(),
        'krg_A': aerial_KRG_A[y_out, x_out].cpu().numpy(),
        'krg_B': aerial_KRG_B[y_out, x_out].cpu().numpy(),
    }


# =============================================================================
# Aggregation across 5 splits
# =============================================================================
def aggregate(results):
    """Compute metrics for each method × frame, averaged across splits."""
    out = {}
    method_to_prefix = {'IDW': 'idw', 'Kriging': 'krg'}
    for method, frame in [('IDW', 'A'), ('IDW', 'B'), ('Kriging', 'A'), ('Kriging', 'B')]:
        prefix = method_to_prefix[method]
        rmses = []
        all_metrics_per_split = []
        for r in results:
            m = compute_metrics(r['true'], r[f"{prefix}_{frame}"])
            rmses.append(m['rmse'])
            all_metrics_per_split.append(m)
        out[f"{method}_{frame}"] = {
            'rmse_mean': float(np.mean(rmses)),
            'rmse_std': float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
            'per_split': [m['rmse'] for m in all_metrics_per_split],
            'mae_mean': float(np.mean([m['mae'] for m in all_metrics_per_split])),
            'pearson_mean': float(np.mean([m['pearson'] for m in all_metrics_per_split])),
            'mbe_mean': float(np.mean([m['mbe'] for m in all_metrics_per_split])),
        }
    return out


# =============================================================================
# Figures
# =============================================================================
def fig_two_frames(agg, out_path):
    methods = ['IDW', 'Kriging']
    frames = ['A', 'B']
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    width = 0.35
    x_pos = np.arange(len(methods))
    a_means = [agg[f"{m}_A"]['rmse_mean'] for m in methods]
    a_stds = [agg[f"{m}_A"]['rmse_std'] for m in methods]
    b_means = [agg[f"{m}_B"]['rmse_mean'] for m in methods]
    b_stds = [agg[f"{m}_B"]['rmse_std'] for m in methods]

    bars_a = ax.bar(x_pos - width / 2, a_means, width, yerr=a_stds, capsize=4,
                    label='Frame A: PSF-applied (surface estimation)', color='#cc4422')
    bars_b = ax.bar(x_pos + width / 2, b_means, width, yerr=b_stds, capsize=4,
                    label='Frame B: PSF-free (aerial interpolation)', color='#3377bb')

    for bar, mean in zip(bars_a, a_means):
        ax.annotate(f'{mean:.0f}', (bar.get_x() + bar.get_width() / 2, mean),
                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)
    for bar, mean in zip(bars_b, b_means):
        ax.annotate(f'{mean:.0f}', (bar.get_x() + bar.get_width() / 2, mean),
                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Held-out RMSE (CPS)')
    ax.set_title('Two evaluation frames: same data, different scientific question\n'
                 '(5 split seeds, 50% held-out)')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    note = ('Note: U-Net is intentionally absent from Frame B.\n'
            'U-Net solves surface estimation, so PSF-free aerial\n'
            'comparison evaluates a task it is not designed for.')
    ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#fffbe6', alpha=0.9))
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight'); plt.close(fig)


def fig_scatter(result_one_split, agg, out_path):
    """Scatter: predicted vs observed for IDW under both frames (single split for clarity)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    panels = [
        (axes[0], result_one_split['idw_A'], 'Frame A: PSF-applied',
         agg['IDW_A']['rmse_mean'], agg['IDW_A']['rmse_std'], '#cc4422'),
        (axes[1], result_one_split['idw_B'], 'Frame B: PSF-free',
         agg['IDW_B']['rmse_mean'], agg['IDW_B']['rmse_std'], '#3377bb'),
    ]
    true = result_one_split['true']
    lim = [0, max(float(true.max()), float(max(panels[0][1].max(), panels[1][1].max())))]
    for ax, vals, lbl, rmse_m, rmse_s, color in panels:
        ax.scatter(true, vals, s=6, alpha=0.5, color=color)
        ax.plot(lim, lim, 'k--', alpha=0.6)
        ax.set_xlabel('Observed CPS')
        ax.set_ylabel('IDW predicted CPS')
        ax.set_title(f"{lbl}\nRMSE = {rmse_m:.0f} ± {rmse_s:.0f} (5-split avg)")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal'); ax.grid(alpha=0.3)
    fig.suptitle(f"IDW under two evaluation frames (split seed {result_one_split['split_seed']})",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight'); plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', type=int, nargs='+', default=[10, 20, 30, 40, 50],
                        help='Split seeds to evaluate (default: 10 20 30 40 50)')
    parser.add_argument('--out_dir', default='results_v81')
    args = parser.parse_args()

    cfg = Config()
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    print(f"Device: {cfg.DEVICE}")
    print(f"Splits: {args.splits}")

    print("\n[1] Loading physics + terrain + data...")
    kernel = get_physics_kernel(cfg, cfg.DEVICE)
    dem, water_mask = load_terrain(cfg)
    df = pd.read_csv(cfg.REAL_CSV_PATH, usecols=['Time', 'Latitude', 'Longitude', 'CPS']).dropna()
    df = df.sort_values(by='Time').reset_index(drop=True)
    print(f"   N = {len(df)},  CPS mean = {df['CPS'].mean():.0f}")

    print("\n[2] Evaluating both frames across splits...")
    results = []
    for ss in args.splits:
        r = evaluate_two_frames(ss, df, cfg, kernel, water_mask)
        m_a = compute_metrics(r['true'], r['idw_A'])
        m_b = compute_metrics(r['true'], r['idw_B'])
        print(f"   split={ss:>2}: IDW Frame A RMSE={m_a['rmse']:>6.1f}  "
              f"Frame B RMSE={m_b['rmse']:>6.1f}  "
              f"ratio={m_a['rmse']/m_b['rmse']:>4.1f}x")
        results.append(r)

    print("\n[3] Aggregating across splits...")
    agg = aggregate(results)

    print("\n" + "=" * 68)
    print("RESULTS  (5 split seeds, 50% held-out, mean ± SD)")
    print("=" * 68)
    print(f"{'':<10} {'Frame A (PSF, surface)':<25} {'Frame B (no PSF, aerial)':<25}")
    print("-" * 68)
    for method in ['IDW', 'Kriging']:
        a = agg[f'{method}_A']
        b = agg[f'{method}_B']
        print(f"{method:<10} RMSE = {a['rmse_mean']:>6.1f} ± {a['rmse_std']:>4.1f}      "
              f"RMSE = {b['rmse_mean']:>6.1f} ± {b['rmse_std']:>4.1f}")
        ratio = a['rmse_mean'] / b['rmse_mean'] if b['rmse_mean'] > 0 else float('inf')
        print(f"{'':<10} (ratio A/B = {ratio:.2f}x — Frame A penalty due to double-blurring)")

    print("\n[4] Generating figures...")
    fig_two_frames(agg, out_dir / "fig_b1_two_frames.png")
    fig_scatter(results[0], agg, out_dir / "fig_b1_scatter.png")
    print(f"   Saved: fig_b1_two_frames.png, fig_b1_scatter.png")

    print("\n[5] Saving JSON...")
    payload = {
        'description': 'Phase B1 supplementary: two evaluation frames',
        'splits': args.splits,
        'aggregated': agg,
        'per_split_raw': [
            {
                'split_seed': r['split_seed'],
                'n_held_out': r['n_held_out'],
                'idw_A_rmse': float(np.sqrt(np.mean((r['true'] - r['idw_A']) ** 2))),
                'idw_B_rmse': float(np.sqrt(np.mean((r['true'] - r['idw_B']) ** 2))),
                'krg_A_rmse': float(np.sqrt(np.mean((r['true'] - r['krg_A']) ** 2))),
                'krg_B_rmse': float(np.sqrt(np.mean((r['true'] - r['krg_B']) ** 2))),
            } for r in results
        ],
    }
    with open(out_dir / "b1_psf_free.json", 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"   Saved: b1_psf_free.json")

    print("\n" + "=" * 68)
    print("PHASE B1 INTERPRETATION (for paper Discussion)")
    print("=" * 68)
    print("""
    Frame A (PSF-applied, main paper):
      IDW/Kriging are interpreted as SURFACE estimates.
      Forward-conv with PSF emulates how an aerial detector would
      observe such a surface; comparison with held-out aerial CPS
      is then meaningful. This is the "double-blurring" frame.

    Frame B (PSF-free, supplementary):
      IDW/Kriging are treated as direct AERIAL interpolators.
      No PSF; comparison is essentially "did the interpolator
      reproduce nearby measurements?" RMSE is much smaller because
      the task is much easier (and largely tautological for IDW).

    The paper's comparison is in Frame A because the operational
    output of contamination mapping is a SURFACE map (used for
    decontamination planning, dose calculation, etc.), not an
    aerial-CPS map. U-Net is designed for Frame A and is therefore
    not evaluated in Frame B.
    """)


if __name__ == '__main__':
    main()
