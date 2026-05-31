"""
Merge two PNG files into a single side-by-side figure with (a) (b) labels.

Used for Fig 6, Fig 7, Fig 8 — each consists of two separate panels
(b3_distribution_overlay + b3_bias_vs_intensity, etc.) that should appear
as a single Figure_N.png in the final submission package.

Workflow on workstation:
    cd radiation-mapping/ukedo_river2
    python tools/make_figures/merge_panels.py

Inputs (from data/results_v81/):
    fig_b3_distribution_overlay.png + fig_b3_bias_vs_intensity.png  -> Figure_6.png
    fig_b2_decile_curve.png         + fig_b2_recovery_rate.png      -> Figure_7.png
    fig_b4_fold_geometry.png        + fig_b4_protocol_comparison.png-> Figure_8.png

    Plus a one-shot copy:
    fig_5x5_rmse_box.png  -> Figure_3.png  (single panel, no merge)

Outputs (to data/results_v81/):
    Figure_3.png    (RMSE boxplot — already single-panel)
    Figure_6.png    (a + b)
    Figure_7.png    (a + b)
    Figure_8.png    (a + b)

Notes:
    - Source PNGs may have built-in titles. We crop the top by a configurable
      ratio (TOP_CROP_FRAC) before placing inside the merged figure to avoid
      duplicate "title" text — the manuscript caption already provides the
      title-equivalent description.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS_DIR = os.path.join(ROOT, 'data', 'results_v81')

# How much of the top of each source PNG to crop
TOP_CROP_FRAC = {
    'fig_b3_distribution_overlay.png': 0.10,
    'fig_b3_bias_vs_intensity.png':    0.13,
    'fig_b2_decile_curve.png':         0.10,
    'fig_b2_recovery_rate.png':        0.10,
    'fig_b4_fold_geometry.png':        0.08,
    'fig_b4_protocol_comparison.png':  0.10,
}


def crop_top(img, frac):
    if frac <= 0:
        return img
    arr = np.asarray(img)
    h = arr.shape[0]
    cut = int(h * frac)
    return Image.fromarray(arr[cut:, :, ...])


def merge_two_panels(left_png, right_png, out_png, label_a='(a)', label_b='(b)',
                     dpi=300):
    """Side-by-side merge with (a)(b) labels."""
    left = Image.open(left_png).convert('RGB')
    right = Image.open(right_png).convert('RGB')
    left = crop_top(left, TOP_CROP_FRAC.get(os.path.basename(left_png), 0.0))
    right = crop_top(right, TOP_CROP_FRAC.get(os.path.basename(right_png), 0.0))

    target_h = left.size[1]
    if right.size[1] != target_h:
        new_w = int(right.size[0] * target_h / right.size[1])
        right = right.resize((new_w, target_h), Image.LANCZOS)

    left_arr = np.asarray(left)
    right_arr = np.asarray(right)

    h_in = 4.0
    w_in_left  = left_arr.shape[1]  / left_arr.shape[0]  * h_in
    w_in_right = right_arr.shape[1] / right_arr.shape[0] * h_in
    total_w_in = w_in_left + w_in_right + 0.1

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(total_w_in, h_in), dpi=dpi,
        gridspec_kw=dict(width_ratios=[w_in_left, w_in_right], wspace=0.02),
    )

    axL.imshow(left_arr, aspect='auto')
    axL.axis('off')
    axR.imshow(right_arr, aspect='auto')
    axR.axis('off')

    axL.text(0.01, 0.99, label_a, transform=axL.transAxes,
             fontsize=14, fontweight='bold', ha='left', va='top',
             family='DejaVu Sans')
    axR.text(0.01, 0.99, label_b, transform=axR.transAxes,
             fontsize=14, fontweight='bold', ha='left', va='top',
             family='DejaVu Sans')

    plt.savefig(out_png, dpi=dpi, bbox_inches='tight', facecolor='white',
                pad_inches=0.05)
    plt.close()
    out_size = os.path.getsize(out_png) / 1024
    print(f"  Saved {os.path.basename(out_png):<20s} ({out_size:>6.1f} KB)")


def copy_single(src_png, out_png):
    import shutil
    shutil.copy(src_png, out_png)
    sz = os.path.getsize(out_png) / 1024
    print(f"  Copied {os.path.basename(out_png):<20s} ({sz:>6.1f} KB)")


def main():
    print("=" * 70)
    print("Panel merge: building Figure_3, _6, _7, _8 PNGs for submission")
    print(f"  Results dir: {RESULTS_DIR}")
    print("=" * 70)

    if not os.path.isdir(RESULTS_DIR):
        raise FileNotFoundError(f"{RESULTS_DIR} not found.")

    print("\nProcessing:")

    # Fig 3 — single panel
    src = os.path.join(RESULTS_DIR, 'fig_5x5_rmse_box.png')
    out = os.path.join(RESULTS_DIR, 'Figure_3.png')
    if os.path.exists(src):
        copy_single(src, out)
    else:
        print(f"  ! Missing: {src}")

    pairs = [
        ('Figure_6.png',
         'fig_b3_distribution_overlay.png',
         'fig_b3_bias_vs_intensity.png'),
        ('Figure_7.png',
         'fig_b2_decile_curve.png',
         'fig_b2_recovery_rate.png'),
        ('Figure_8.png',
         'fig_b4_fold_geometry.png',
         'fig_b4_protocol_comparison.png'),
    ]
    for out_name, a_name, b_name in pairs:
        src_a = os.path.join(RESULTS_DIR, a_name)
        src_b = os.path.join(RESULTS_DIR, b_name)
        out = os.path.join(RESULTS_DIR, out_name)
        if os.path.exists(src_a) and os.path.exists(src_b):
            merge_two_panels(src_a, src_b, out)
        else:
            missing = [x for x in (src_a, src_b) if not os.path.exists(x)]
            print(f"  ! Missing source(s): {missing}")

    print()
    print("Submission-ready figures:")
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.startswith('Figure_') and fname.endswith('.png'):
            full = os.path.join(RESULTS_DIR, fname)
            sz = os.path.getsize(full) / 1024
            print(f"  {fname:<20s}  {sz:>7.1f} KB")


if __name__ == '__main__':
    main()
