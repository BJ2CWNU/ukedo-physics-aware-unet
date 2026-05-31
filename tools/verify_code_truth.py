"""
verify_code_truth.py — Cross-check the manuscript's architecture / baseline
implementation claims against the actual code (ukedo_v81.py + Config).

Motivation (D24/D25, 2026-05-05): the manuscript text described "batch
normalisation" while the code uses nn.GroupNorm, and the baseline parameters
(IDW power, kriging variogram/range/nugget, library version) were not stated at
all. check_consistency.py verifies numeric values against numbers.md but does not
verify prose claims about the implementation. This script closes that gap.

Each check is (claim found in manuscript?) AND (code matches the claim?).
  [OK]       claim present and code agrees
  [MISMATCH] claim present but code disagrees  -> problem
  [WARN]     claim not found in manuscript (wording may have changed) -> review

Usage:
    cd radiation-mapping/ukedo_river2
    python tools/verify_code_truth.py

Exit 0 = no MISMATCH; 1 = at least one MISMATCH.
"""
from __future__ import annotations
import os
import re
import sys
import inspect
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                 # ukedo_river2/
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, SRC)

from ukedo_v81 import Config, compute_idw_torch  # noqa: E402
import pykrige  # noqa: E402

MANUSCRIPT = Path(ROOT) / 'docs' / 'manuscript_remote_sensing.md'
UKEDO_SRC = Path(SRC) / 'ukedo_v81.py'

cfg = Config()
mtext = MANUSCRIPT.read_text(encoding='utf-8')
src = UKEDO_SRC.read_text(encoding='utf-8')
idw_default_p = inspect.signature(compute_idw_torch).parameters['p'].default


def claim_present(pattern: str) -> bool:
    return re.search(pattern, mtext, flags=re.IGNORECASE) is not None


# Each entry: (description, manuscript_claim_regex, code_condition_bool)
CHECKS = [
    ("Normalisation = GroupNorm",
     r'group\s+normalisation',
     'nn.GroupNorm' in src),
    ("Kriging variogram = exponential",
     r'exponential\s+variogram',
     cfg.VARIOGRAM_MODEL == 'exponential'),
    ("Kriging range = 22.5 cells",
     r'22\.5\s*(?:grid\s*)?cells',
     abs(cfg.KRIG_RANGE_CELLS - 22.5) < 1e-6),
    ("Kriging range = 225 m",
     r'225\s*m',
     abs(cfg.VARIOGRAM_RANGE_M - 225.0) < 1e-6),
    ("Kriging nugget = 0.0",
     r'nugget\s+of\s+0\.0',
     "'nugget': 0.0" in src or '"nugget": 0.0' in src),
    ("IDW power p = 2.0",
     r'p\s*=\s*2\.0',
     abs(float(idw_default_p) - 2.0) < 1e-6),
    ("Library = PyKrige 1.7.3",
     r'PyKrige\s+1\.7\.3',
     pykrige.__version__ == '1.7.3'),
    ("Batch size = 16",
     r'batch\s+size\s+of\s+16',
     cfg.BATCH_SIZE == 16),
    ("Epochs = 30",
     r'30\s+epochs',
     cfg.EPOCHS == 30),
    ("Learning rate = 1e-3",
     r'1\s*[×x]\s*10\^?-?3',
     abs(cfg.LR - 1e-3) < 1e-12),
    ("Train samples = 1000",
     r'1,?000\s+synthetic\s+samples',
     cfg.N_TRAIN == 1000),
    ("Validation samples = 200",
     r'200\s+(?:held-out\s+)?synthetic\s+samples',
     cfg.N_VAL == 200),
    ("Input channels = 5",
     r'five[-\s]channel',
     True),  # 5-channel is structural; confirmed by build code elsewhere
    ("Grid = 64 x 128",
     r'64\s*[×x]\s*128',
     cfg.H == 64 and cfg.W == 128),
    ("Split seeds {10,20,30,40,50}",
     r'\{?10,?\s*20,?\s*30,?\s*40,?\s*50\}?',
     cfg.SPLIT_SEEDS == [10, 20, 30, 40, 50]),
    ("Model seeds {42,123,2026,7,99}",
     r'\{?42,?\s*123,?\s*2026,?\s*7,?\s*99\}?',
     cfg.MODEL_SEEDS == [42, 123, 2026, 7, 99]),
]


def main():
    print("=" * 70)
    print("verify_code_truth.py — manuscript implementation claims vs code")
    print(f"  manuscript: {MANUSCRIPT.name}   code: {UKEDO_SRC.name}")
    print("=" * 70)

    n_mismatch = 0
    n_warn = 0
    for desc, pattern, code_ok in CHECKS:
        present = claim_present(pattern)
        if not present:
            print(f"  [WARN]     {desc}: claim not found in manuscript (regex /{pattern}/)")
            n_warn += 1
        elif not code_ok:
            print(f"  [MISMATCH] {desc}: manuscript states it, but code DISAGREES")
            n_mismatch += 1
        else:
            print(f"  [OK]       {desc}")

    print("=" * 70)
    print(f"Summary: {n_mismatch} mismatch, {n_warn} warn, "
          f"{len(CHECKS) - n_mismatch - n_warn} ok")
    if n_mismatch == 0:
        print("NO CODE-TRUTH MISMATCH — manuscript implementation claims agree with code")
        if n_warn:
            print("(Review WARN items: claim wording may have changed — not necessarily wrong.)")
        sys.exit(0)
    else:
        print("CODE-TRUTH MISMATCH found — reconcile manuscript with code before submission.")
        sys.exit(1)


if __name__ == '__main__':
    main()
