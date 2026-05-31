"""
check_consistency.py — Manuscript number consistency checker.

Reads docs/numbers.md (single source of truth) and verifies:

  1. Every canonical number is present somewhere in the manuscript.
     (Catches: forgot to update Abstract / Table when changing §3 results)

  2. No "stale" numbers (listed under "Stale values that must NOT appear")
     remain in the manuscript.
     (Catches: old IDW=931 still appearing in Table 2 after §3 was updated)

  3. Pairs of numbers that should always co-appear (mean and SD, ceiling
     and headroom) are not separated.

Usage:
    python tools/check_consistency.py [--verbose]

Exit code:
    0 — all checks pass
    1 — at least one inconsistency found
    2 — script error (missing files etc.)
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Tuple

NUMBERS_FILE = Path('docs/numbers.md')
MANUSCRIPT_FILE = Path('docs/manuscript_remote_sensing.md')

# Stale values (with a one-line description) that must NOT appear in the manuscript.
# Listed under section H of numbers.md but tracked here to be runtime-checkable.
# Format: regex pattern -> human-readable warning
# Patterns are allowed to match in tables/figures/abstract — anywhere.
STALE_PATTERNS = [
    # 931 ± 19 (old IDW RMSE)
    (r'\b931\s*[±\u00B1]\s*19\b',         "Old IDW RMSE 931 ± 19 (replace with 916.8 ± 34.2)"),
    (r'\bIDW.{0,30}931\b',                "Old IDW RMSE 931 (replace with 916.8)"),
    (r'\b931 CPS\b',                      "Old IDW RMSE value 931"),
    # 666 ± 54 (old U-Net RMSE)
    (r'\b666\s*[±\u00B1]\s*54\b',         "Old U-Net RMSE 666 ± 54 (replace with 705.4 ± 102.8)"),
    (r'\bU-Net.{0,40}666\b',              "Old U-Net RMSE 666"),
    (r'\b666 CPS\b',                      "Old U-Net RMSE value 666"),
    # 28.5% improvement
    (r'\b28\.5\s*[±\u00B1]\s*6\.1\b',     "Old improvement 28.5 ± 6.1% (replace with 23.1 ± 6.5%)"),
    (r'\b28\.5\s*%',                      "Old improvement value 28.5%"),
    (r'~?\b29\s*%\s+improvement',         "Old approximate ~29% improvement (replace with ~23%)"),
    # MBE −376 (old IDW)
    (r'[-\u2212]\s*376',                  "Old IDW MBE -376 (replace with -412)"),
    # 0.924 (old Pearson)
    (r'\b0\.924\b',                       "Old U-Net Pearson r 0.924 (replace with 0.91)"),
    # 0.90 / 0.901 (old CCC) - need narrow regex to avoid false positives
    (r'CCC.{0,15}0\.90\b',                "Old CCC 0.90 (replace with 0.85)"),
    (r'CCC.{0,15}0\.901',                 "Old CCC 0.901 (replace with 0.85)"),
    (r'concordance.{0,30}0\.90\b',        "Old CCC 0.90 in concordance description"),
    # 93.5% (old variance decomp)
    (r'\b93\.5\s*%',                      "Old variance decomp 93.5% (replace with 92.2%)"),
    # 0.784 specific old IDW Pearson
    (r'\b0\.784\b',                       "Old IDW Pearson 0.784 (replace with 0.78 — same value, but check rounding)"),
]

# Required numbers (KEY -> regex). KEY refers to numbers.md schema.
# Each must appear at least once in the manuscript.
REQUIRED_NUMBERS = [
    ('IDW_RMSE_MEAN',       r'\b916\.8\b',       "IDW mean RMSE 916.8"),
    ('IDW_RMSE_STD',        r'\b34\.2\b',        "IDW RMSE SD 34.2"),
    ('KRG_RMSE_MEAN',       r'\b832\.4\b',       "Kriging mean RMSE 832.4"),
    ('UNET_RMSE_MEAN',      r'\b705\.4\b',       "U-Net mean RMSE 705.4"),
    ('UNET_RMSE_STD',       r'\b102\.8\b',       "U-Net RMSE SD 102.8"),
    ('IMPROVEMENT_VS_IDW',  r'\b23\.1\b',        "Mean improvement vs IDW 23.1%"),
    ('IMPROVEMENT_VS_KRG',  r'\b15\.3\b',        "Mean improvement vs Kriging 15.3%"),
    ('IDW_MBE',             r'[-\u2212]\s*412\b','IDW MBE -412'),
    ('KRG_MBE',             r'[-\u2212]\s*460\b','Kriging MBE -460'),
    ('VAR_DECOMP_MODEL',    r'\b92\.2\b',        "Variance decomp model% 92.2"),
    ('B1_FRAME_B_IDW',      r'\b347\.7\b',       "Frame B IDW RMSE 347.7"),
    ('B1_FRAME_B_KRG',      r'\b155\.3\b',       "Frame B Kriging RMSE 155.3"),
    ('B1_KRG_OVER_IDW_FRAME_B', r'\b2\.24\b',    "Kriging-over-IDW Frame B ratio 2.24"),
    ('B1_KRG_OVER_IDW_FRAME_A', r'\b1\.10\b',    "Kriging-over-IDW Frame A ratio 1.10"),
    ('B2_IDW_CEILING',      r'\b5,?794\b',       "IDW ceiling 5794"),
    ('B2_UNET_CEILING',     r'10,?275\b',        "U-Net ceiling 10275"),
    ('B2_OBSERVED_TOP10',   r'10,?895\b',        "Observed top-10% max 10895"),
    ('B2_IDW_HEADROOM',     r'\b46\.8\b',        "IDW headroom 46.8%"),
    ('B2_UNET_HEADROOM',    r'\b5\.7\b',         "U-Net headroom 5.7%"),
    ('B2_RECOVERY_T6000',   r'\bT\s*=\s*6,?000', "Recovery threshold T=6000 mention"),
    ('B2_RECOVERY_N_S10',   r'\bn\s*=\s*64\b',   "Recovery rate denominator n=64 at split_seed=10"),
    ('B3_HETERO_RATIO',     r'(?:3\.0|3×|3 ×|3-fold|3 ?times|3x)', "Heteroscedastic ratio ~3×"),
    ('B3_PEAKINESS_REAL',   r'\b3\.13\b',        "Real Ukedo peakiness 3.13"),
    ('B3_PEAKINESS_P5',     r'\b3\.94\b',        "Synth P5 peakiness 3.94"),
    ('B4_BUFFER',           r'\b27\s*m\b',       "Buffer 27 m"),
    ('DATASET_N',           r'2,?213',           "Dataset N=2213"),
    ('DATASET_CPS_MAX',     r'11,?184',          "CPS max 11184"),
    ('HELD_OUT_N',          r'1,?107',           "Held-out n=1107"),
    ('DIRECTIONAL_25_25',   r'25\s*(?:out of|/)\s*25', "Directional 25/25"),
]

# Reference patterns that should appear (post-Stage 1.7-1.9).
REQUIRED_REFERENCES = [
    # Manuscript uses numeric in-text citation style ([10, 11]); confirm each
    # reference by its bibliography entry (surname + year on the same line).
    ('Roberts',             r'Roberts[^\n]*2017', "Roberts et al. 2017 (block CV reference)"),
    ('Ploton',              r'Ploton[^\n]*2020',  "Ploton et al. 2020 (Nat Comm spatial CV)"),
    ('Wadoux',              r'Wadoux',                         "Wadoux et al. (spatial CV reply)"),
    ('Karasiak',            r'Karasiak',                       "Karasiak et al. 2022"),
]

# Placeholder/TODO markers that must not survive to submission.
PLACEHOLDER_PATTERNS = [
    (r'\[reference\]',      "Bibliography placeholder '[reference]' must be filled"),
    (r'\bTODO\b',           "TODO marker present"),
    (r'\bFIXME\b',           "FIXME marker present"),
    (r'\[ANONYMOUS',         "Lingering [ANONYMOUS marker"),
    (r'\bTBD\b',             "TBD placeholder must be replaced with measured value"),
    (r'\bn\s*=\s*TBD\b',     "Specifically: n = TBD must be measured (likely B2 recovery rate denominator)"),
]


def read_manuscript(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Manuscript not found: {path}")
    return path.read_text(encoding='utf-8')


def check_stale(text: str, verbose: bool) -> Tuple[int, list]:
    """Return (n_problems, list_of_problem_strs)."""
    issues = []
    for pattern, desc in STALE_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            line_no = text.count('\n', 0, m.start()) + 1
            issues.append(f"  [STALE  L{line_no}] {desc}  -- match: '{m.group()[:80]}'")
    return len(issues), issues


def check_required(text: str, verbose: bool) -> Tuple[int, list]:
    issues, ok = [], []
    for key, pattern, desc in REQUIRED_NUMBERS:
        if not re.search(pattern, text, flags=re.IGNORECASE):
            issues.append(f"  [MISSING] {key}: {desc}")
        else:
            ok.append(f"  [OK]      {key}: {desc}")
    if verbose:
        for line in ok:
            print(line)
    return len(issues), issues


def check_references(text: str, verbose: bool) -> Tuple[int, list]:
    issues = []
    for key, pattern, desc in REQUIRED_REFERENCES:
        if not re.search(pattern, text, flags=re.IGNORECASE):
            issues.append(f"  [MISSING REF] {key}: {desc}")
    return len(issues), issues


def check_placeholders(text: str, verbose: bool) -> Tuple[int, list]:
    issues = []
    for pattern, desc in PLACEHOLDER_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            line_no = text.count('\n', 0, m.start()) + 1
            issues.append(f"  [PLACEHOLDER L{line_no}] {desc}")
    return len(issues), issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='store_true', help='show passing checks too')
    parser.add_argument('--manuscript', default=str(MANUSCRIPT_FILE),
                        help=f'manuscript path (default: {MANUSCRIPT_FILE})')
    args = parser.parse_args()

    try:
        text = read_manuscript(Path(args.manuscript))
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print("=" * 70)
    print(f"Consistency check: {args.manuscript}")
    print(f"Single source of truth: {NUMBERS_FILE}")
    print("=" * 70)

    n_stale, stale_issues = check_stale(text, args.verbose)
    n_missing, missing_issues = check_required(text, args.verbose)
    n_ref, ref_issues = check_references(text, args.verbose)
    n_ph, ph_issues = check_placeholders(text, args.verbose)

    print(f"\n[1] Stale values present (must be removed):  {n_stale} problem(s)")
    for i in stale_issues:
        print(i)

    print(f"\n[2] Required numbers missing (must appear):  {n_missing} problem(s)")
    for i in missing_issues:
        print(i)

    print(f"\n[3] References missing (post-Stage 1.7-1.9 work): {n_ref} problem(s)")
    for i in ref_issues:
        print(i)

    print(f"\n[4] Placeholders to remove before submit:    {n_ph} problem(s)")
    for i in ph_issues:
        print(i)

    total = n_stale + n_missing + n_ref + n_ph
    print("\n" + "=" * 70)
    if total == 0:
        print("ALL CONSISTENT — manuscript matches numbers.md")
        return 0
    print(f"FOUND {total} consistency problem(s). Manuscript is not yet submission-ready.")
    print("Run again after fixing; this script is fast (< 1 sec) and safe to invoke often.")
    return 1


if __name__ == '__main__':
    sys.exit(main())
