"""
verify_numbers.py — Recompute numbers.md statistics from the actual code/corpus
and flag any stale value (e.g. a 200->1000 sample partial-update residue).

Motivation (D24, 2026-05-05): the B3 P95 peakiness value in numbers.md was stale
(7.55 from an old 200-sample measurement) while P50 had been updated to the
1000-sample value (5.18). check_consistency.py only verifies that the manuscript
matches numbers.md — it cannot catch numbers.md itself being wrong. This script
closes that gap by regenerating the synthetic corpus statistics deterministically
and comparing against the claimed values in numbers.md.

Usage:
    cd radiation-mapping/ukedo_river2
    python tools/verify_numbers.py

Exit code 0 = all values match (within tolerance); 1 = at least one stale value.
"""
from __future__ import annotations
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                 # ukedo_river2/
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, SRC)
# load_terrain reads 'ukedo_water_mask.png' relative to cwd -> chdir to data/
os.chdir(os.path.join(ROOT, 'data'))

from ukedo_v81 import Config, get_physics_kernel, load_terrain, make_ground_source  # noqa: E402

NUMBERS_MD = Path(ROOT) / 'docs' / 'numbers.md'
N_SYNTH = 1000            # must match numbers.md "Synthetic training distribution (N samples)"
CORPUS_SEED = 12345       # Config.TRAIN_DATA_SEED
REAL_PEAKINESS = 3.13     # threshold for frac_below_real (matches numbers.md B3_REAL_PEAKINESS)
TOL = 0.01                # absolute tolerance for percentile match
TOL_FRAC = 0.002          # absolute tolerance for fraction match


def measure_corpus(cfg, water_mask, n=N_SYNTH, seed=CORPUS_SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    ratios = []
    for _ in range(n):
        gt = make_ground_source(cfg, water_mask).cpu().numpy()
        if gt.mean() > 0:
            ratios.append(float(gt.max() / gt.mean()))
    r = np.array(ratios)
    return {
        'B3_SYNTH_PEAKINESS_P5':  float(np.percentile(r, 5)),
        'B3_SYNTH_PEAKINESS_P50': float(np.percentile(r, 50)),
        'B3_SYNTH_PEAKINESS_P95': float(np.percentile(r, 95)),
        'B3_FRAC_SYNTH_BELOW_REAL': float(np.mean(r < REAL_PEAKINESS)),
    }


def parse_numbers_md(path: Path):
    text = path.read_text(encoding='utf-8')
    claimed = {}
    for key in ('B3_SYNTH_PEAKINESS_P5', 'B3_SYNTH_PEAKINESS_P50',
                'B3_SYNTH_PEAKINESS_P95', 'B3_FRAC_SYNTH_BELOW_REAL'):
        m = re.search(rf'{re.escape(key)}\s*=\s*([0-9.]+)', text)
        if m:
            claimed[key] = float(m.group(1))
    return claimed


def main():
    cfg = Config()
    _ = get_physics_kernel(cfg, cfg.DEVICE)   # not needed for peakiness, kept for parity
    dem, water_mask = load_terrain(cfg)

    print("=" * 70)
    print(f"verify_numbers.py — recomputing B3 peakiness from {N_SYNTH}-sample corpus")
    print(f"  seed={CORPUS_SEED}  device={cfg.DEVICE}")
    print("=" * 70)

    actual = measure_corpus(cfg, water_mask)
    claimed = parse_numbers_md(NUMBERS_MD)

    n_problems = 0
    for key, act in actual.items():
        tol = TOL_FRAC if 'FRAC' in key else TOL
        if key not in claimed:
            print(f"  [MISSING]  {key}: not found in numbers.md (actual={act:.3f})")
            n_problems += 1
            continue
        clm = claimed[key]
        if abs(act - clm) > tol:
            print(f"  [STALE]    {key}: numbers.md claims {clm:.3f}, actual {act:.3f}  (|Δ|={abs(act-clm):.3f} > {tol})")
            n_problems += 1
        else:
            print(f"  [OK]       {key}: {clm:.3f} == {act:.3f}")

    print("=" * 70)
    if n_problems == 0:
        print("ALL NUMBERS CONSISTENT — numbers.md matches the regenerated corpus")
        sys.exit(0)
    else:
        print(f"FOUND {n_problems} stale/missing value(s). Update numbers.md (and re-run check_consistency.py).")
        sys.exit(1)


if __name__ == '__main__':
    main()
