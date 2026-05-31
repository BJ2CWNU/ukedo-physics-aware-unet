"""
Fig 2 — Two-phase sim-to-real workflow diagram.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(ROOT, 'data', 'results_v81')
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, 'Figure_2.png')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8.0,
    'axes.linewidth': 0.7,
})

GREEN_FILL  = '#E8F5E9'
GREEN_EDGE  = '#2E7D32'
AMBER_FILL  = '#FFF8E1'
AMBER_EDGE  = '#F57C00'
BLUE_FILL   = '#E3F2FD'
BLUE_EDGE   = '#1565C0'
GREY_FILL   = '#F5F5F5'
GREY_EDGE   = '#424242'
RED_EDGE    = '#C62828'

fig_w_in = 7.0
fig_h_in = 5.5
fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')


def box(x, y, w, h, label, fill, edge, fontsize=8.0, fontweight='normal',
        italic=False, alpha=1.0):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.3,rounding_size=0.5",
        linewidth=0.9, edgecolor=edge, facecolor=fill, alpha=alpha,
    )
    ax.add_patch(rect)
    style_kw = {'fontsize': fontsize, 'fontweight': fontweight}
    if italic:
        style_kw['fontstyle'] = 'italic'
    ax.text(x + w / 2, y + h / 2, label,
            ha='center', va='center', **style_kw)


def arrow(x1, y1, x2, y2, color='#424242', lw=1.0, style='-|>',
          linestyle='solid'):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
    )
    ax.add_patch(arr)


def phase_band(y, h, fill, edge, label, label_color):
    rect = patches.Rectangle((1, y), 98, h, facecolor=fill, edgecolor=edge,
                             linewidth=0.6, alpha=0.35)
    ax.add_patch(rect)
    ax.text(2.5, y + h - 1.5, label, fontsize=9, fontweight='bold',
            color=label_color, ha='left', va='top', style='italic')


# ============================================================================
# PRETRAINING phase (top, y=55-99)
# ============================================================================
phase_band(55, 44, GREEN_FILL, GREEN_EDGE,
           'Pretraining (synthetic data only)', GREEN_EDGE)

# Top row: synthetic data pipeline
box(4, 86, 14, 8, 'Synthetic\nground source\nC(x, y)', BLUE_FILL, BLUE_EDGE,
    fontsize=7.5)
box(23, 87, 7, 6, 'K *', GREY_FILL, GREY_EDGE, italic=True)
arrow(18, 90, 23, 90, color=GREY_EDGE, lw=0.9)
box(35, 86, 14, 8, 'Synthetic\naerial\nA = C * K', BLUE_FILL, BLUE_EDGE,
    fontsize=7.5)
arrow(30, 90, 35, 90, color=GREY_EDGE, lw=0.9)
box(54, 86, 12, 8, 'Random\nmask\n(10–90%)', BLUE_FILL, BLUE_EDGE,
    fontsize=7.4)
arrow(49, 90, 54, 90, color=GREY_EDGE, lw=0.9)

# 5-channel input box (right top, compact title only)
box(72, 86, 24, 7, '5-channel input', BLUE_FILL, BLUE_EDGE,
    fontweight='bold', fontsize=8.5)
arrow(66, 89.5, 72, 89.5, color=GREY_EDGE, lw=0.9)

# Channel labels OUTSIDE the 5-channel box (separated, readable)
ax.text(84, 84.5,
        '(1) sparse aerial  (2) IDW baseline  (3) land-water scalar\n(4) measurement mask  (5) water mask',
        ha='center', va='center', fontsize=6.5, color=GREY_EDGE,
        family='DejaVu Sans', style='italic')

# Middle row: U-Net + outputs (right-to-left flow)
# Lengthen the arrow for clear visual separation from 5-channel box
arrow(84, 80, 84, 79, color=GREY_EDGE, lw=0.9)
box(70, 71, 28, 8, 'Physics-aware U-Net (encoder–decoder)',
    GREY_FILL, GREEN_EDGE, fontsize=8.2, fontweight='bold')
arrow(70, 75, 65, 75, color=GREEN_EDGE, lw=0.9)
box(50, 71, 14, 8, 'Predicted\nground Ĉ', GREEN_FILL, GREEN_EDGE,
    fontsize=7.5)
arrow(50, 75, 45, 75, color=GREY_EDGE, lw=0.9)
box(38, 72, 7, 6, 'K *', GREY_FILL, GREY_EDGE, italic=True)
arrow(38, 75, 33, 75, color=GREY_EDGE, lw=0.9)
box(18, 71, 14, 8, 'Predicted\naerial Â', GREEN_FILL, GREEN_EDGE,
    fontsize=7.5)

# Composite loss
box(18, 58, 60, 9,
    'Composite loss (SmoothL1): surface (Ĉ vs C) + 2.0 × aerial (Â vs A)',
    '#FFEBEE', RED_EDGE, fontsize=8.0, fontweight='bold')
arrow(57, 71, 48, 67, color=RED_EDGE, lw=0.9)
arrow(25, 71, 30, 67, color=RED_EDGE, lw=0.9)

# ============================================================================
# Transition: Frozen weights badge (between phases)
# ============================================================================
box(80, 46, 18, 9,
    'Trained weights\nFROZEN',
    GREEN_FILL, GREEN_EDGE, fontweight='bold', fontsize=8.0)
arrow(84, 71, 89, 55, color=GREEN_EDGE, lw=1.4, style='-|>')

# ============================================================================
# EVALUATION phase (bottom, y=2-44)
# ============================================================================
phase_band(2, 42, AMBER_FILL, AMBER_EDGE,
           'Evaluation (real Ukedo data, no fine-tuning)', AMBER_EDGE)

box(4, 30, 16, 8, 'Real Ukedo\ntrajectory\n2,213 points', BLUE_FILL,
    BLUE_EDGE, fontsize=7.4)
box(24, 30, 14, 8, '50% split', BLUE_FILL, BLUE_EDGE, fontsize=7.6)
arrow(20, 34, 24, 34, color=GREY_EDGE, lw=0.9)
box(42, 30, 18, 8,
    'Single IDW\ninterpolation',
    AMBER_FILL, AMBER_EDGE, fontsize=7.6, fontweight='bold')
arrow(38, 34, 42, 34, color=GREY_EDGE, lw=0.9)

ax.text(51, 26, '(IDW used in two roles ↓)',
        ha='center', va='top', fontsize=7.0, style='italic',
        color=AMBER_EDGE)

# Role 1: IDW baseline surface proxy
box(64, 33, 14, 8, 'Role 1:\nIDW baseline\nsurface proxy', AMBER_FILL,
    AMBER_EDGE, fontsize=6.8)
arrow(60, 35, 64, 36, color=AMBER_EDGE, lw=0.9)

# Role 2: Channel 1 of 5-ch input
box(64, 21, 14, 8, 'Role 2: Ch 1 of\n5-ch input → U-Net',
    AMBER_FILL, AMBER_EDGE, fontsize=6.8)
arrow(60, 33.5, 64, 26, color=AMBER_EDGE, lw=0.9)

# Frozen U-Net (eval inference)
box(82, 21, 14, 8, 'Frozen U-Net\ninference', GREY_FILL, GREEN_EDGE,
    fontsize=7.4, fontweight='bold')
arrow(78, 25, 82, 25, color=GREEN_EDGE, lw=0.9)
# Connect frozen weights badge (top) → frozen U-Net (bottom) — dashed
arrow(89, 46, 89, 29, color=GREEN_EDGE, lw=1.0, style='-|>',
      linestyle='dashed')

# Predicted ground (eval)
box(82, 11, 14, 8, 'Predicted\nground Ĉ\n→ K * → Â',
    GREEN_FILL, GREEN_EDGE, fontsize=6.8)
arrow(89, 21, 89, 19, color=GREEN_EDGE, lw=0.9)

# Forward project IDW (Role 1)
arrow(78, 37, 82, 37, color=GREY_EDGE, lw=0.9)
box(82, 33, 14, 8, 'IDW surface\n→ K * → Â',
    AMBER_FILL, AMBER_EDGE, fontsize=6.8)

# Score on held-out
box(20, 5, 60, 7,
    'Score at 1,107 held-out points (25 AI runs, 5 IDW runs)',
    '#FFEBEE', RED_EDGE, fontsize=8.0, fontweight='bold')
arrow(85, 33, 50, 12, color=RED_EDGE, lw=0.9)
arrow(85, 11, 50, 11.5, color=RED_EDGE, lw=0.9)
arrow(31, 30, 31, 12, color=RED_EDGE, lw=0.9)

plt.tight_layout(pad=0.4)
plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved: {OUT}")
