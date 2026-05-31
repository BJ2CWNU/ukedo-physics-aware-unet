"""
measure_n_at_threshold.py — One-shot measurement to fill Table 2b's "n = TBD".

Reports the number of held-out points whose observed CPS exceeds T = 6,000 CPS.
Uses CSV only — no model inference, no GPU. Runs in ~1 second on any CPU.

This script answers the question Table 2b leaves open:
    "Recovery rate at T = 6,000 CPS (n = TBD held-out points with obs >= T)"

Strategy:
  1. Load full dataset (n=2213 expected)
  2. For each of 5 splits used in the manuscript (split_seeds 10, 20, 30, 40, 50),
     reconstruct the same 50% holdout split as the manuscript
  3. Count how many held-out points have CPS >= 6,000 in each split
  4. Report mean across splits (this is what "n" should be in Table 2b)

Usage:
  cd ukedo_river2
  python tools/measure_n_at_threshold.py

Output:
  Prints n at T=6000 across the 5 splits, plus the canonical value to put
  in Table 2b. No file modification — manual edit guided by output.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Locate CSV — prefer Phase A4 path (data/), fall back to repo
CSV_CANDIDATES = [
    Path('data/integrated data.csv'),
    Path('integrated data.csv'),
    Path('../data/integrated data.csv'),
]
csv_path = next((p for p in CSV_CANDIDATES if p.is_file()), None)
if csv_path is None:
    print("ERROR: integrated data.csv not found in any of:", file=sys.stderr)
    for p in CSV_CANDIDATES:
        print(f"  - {p.resolve()}", file=sys.stderr)
    sys.exit(2)

THRESHOLD = 6000
SPLIT_SEEDS = [10, 20, 30, 40, 50]
HOLDOUT_RATIO = 0.5

print(f"CSV: {csv_path.resolve()}")
df = pd.read_csv(csv_path, usecols=['Time', 'Latitude', 'Longitude', 'CPS']).dropna()
df = df.sort_values(by='Time').reset_index(drop=True)
n_total = len(df)
n_total_above = int((df['CPS'] >= THRESHOLD).sum())
print(f"Dataset: n = {n_total} total points")
print(f"Total points with CPS >= {THRESHOLD}: {n_total_above}  ({100*n_total_above/n_total:.1f}% of dataset)")
print()
print(f"Per-split holdout sizes (matches Phase A4 split_real_data convention):")
print()

# Reconstruct exact same split logic as ukedo_v81.split_real_data
counts = []
for seed in SPLIT_SEEDS:
    rng = np.random.RandomState(seed)
    n_holdout = int(n_total * HOLDOUT_RATIO)
    perm = rng.permutation(n_total)
    holdout_idx = perm[:n_holdout]
    holdout_cps = df['CPS'].iloc[holdout_idx].values
    n_above = int((holdout_cps >= THRESHOLD).sum())
    counts.append(n_above)
    print(f"  split_seed={seed:3d}: n_holdout={n_holdout}, n(CPS >= {THRESHOLD}) = {n_above}")

mean_n = float(np.mean(counts))
sd_n = float(np.std(counts, ddof=1))
print()
print(f"  Across 5 splits: n = {mean_n:.0f} +/- {sd_n:.0f}")
print(f"  Range: [{min(counts)}, {max(counts)}]")
print()
print("=" * 70)
print("CANONICAL VALUE FOR TABLE 2b")
print("=" * 70)
print()
print(f"  n at T = {THRESHOLD} CPS = {round(mean_n)}")
print(f"  (mean across 5 split seeds, range {min(counts)}-{max(counts)})")
print()
print("Update locations:")
print(f"  1. docs/manuscript_remote_sensing.md, Table 2b row:")
print(f"       Recovery rate at T = 6,000 CPS (n = {round(mean_n)} held-out points...")
print(f"  2. docs/numbers.md, add to Phase B2 section:")
print(f"       B2_RECOVERY_T6000_N = {round(mean_n)}  [points]  # mean across 5 splits")
print(f"  3. Then: python tools/check_consistency.py")
print(f"       expected: 0 placeholders (TBD removed)")
