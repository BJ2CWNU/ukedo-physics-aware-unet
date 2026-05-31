# ukedo-physics-aware-unet

Code and processed data for:

> Kim, B.-J. (2026). Physics-aware deep learning reconstructs ground contamination from sparse UAV radiation measurements over the Fukushima Ukedo basin without field training. *Remote Sensing*, in review.

## Overview

This repository contains the reproducible implementation of a physics-aware
U-Net for reconstructing ground-level radiation contamination from sparse
unmanned aerial vehicle (UAV) measurements. The model is pretrained entirely
on synthetic data and deployed without field fine-tuning on Fukushima Ukedo
basin trajectory data originally surveyed by Kim et al. (2019).

**Key results** (50% within-system random holdout, 25 split × initialisation runs):

| Metric | IDW | Ordinary Kriging | Physics-aware U-Net |
| --- | --- | --- | --- |
| RMSE (CPS) | 916.8 ± 34.2 | 832.4 ± 31.3 | **705.4 ± 102.8** |
| Pearson r † | 0.78 | 0.85 | **0.91** |
| Hotspot recovery at T=6000 CPS | ~0% | ~0% | **~80%** |
| Directional improvement vs IDW | — | 5/5 | **25/25** |

† Pearson r and CCC reported for the same predictions: deterministic IDW/kriging and the 3-model U-Net ensemble at split seed 10.

## Repository Structure

```
.
├── src/                Pipeline implementation
│   ├── ukedo_v81.py                  PhysicsAwareUNet + training + evaluation
│   ├── compute_idw_torch.py          IDW baseline (pure PyTorch, p=2.0)
│   ├── compute_kriging.py            PyKrige baseline (exponential variogram)
│   └── b3_distribution_mismatch.py   Fig 6(a) generation
│
├── tools/              Automated consistency verification
│   ├── check_consistency.py          text ↔ numbers.md
│   ├── verify_numbers.py             numbers.md ↔ corpus recomputation
│   ├── verify_code_truth.py          manuscript prose ↔ code implementation
│   └── insert_figures.py             docx figure embedding
│
├── figures/            Manuscript figures + generation scripts
│   ├── Figure_1.png .. Figure_8.png
│   └── scripts/
│
├── data/               Processed data
│   ├── processed/                    Ukedo trajectory + split indices
│   └── synthetic/                    Corpus statistics (P5/P50/P95 verification)
│
├── results/            Per-run result tables underlying Tables 2a, 2b, 3
│
└── docs/               Source-of-truth documents
    ├── numbers.md
    └── design_decisions.md
```

## Reproducing the Reported Results

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Pretrain U-Net on the synthetic corpus

```bash
python src/ukedo_v81.py --mode pretrain
```

Uses fixed split seeds {10, 20, 30, 40, 50} and model seeds {42, 123, 2026, 7, 99}.

### 3. Evaluate on Ukedo held-out data

```bash
python src/ukedo_v81.py --mode evaluate
```

Produces the 25 single-model runs (5 splits × 5 model seeds) for Table 2a,
and the 3-model ensemble at split seed 10 for Table 2b.

### 4. Generate manuscript figures

```bash
python figures/scripts/make_fig2.py
python figures/scripts/predict_split10_seed42.py
python figures/scripts/plot_fig4.py
python figures/scripts/plot_fig5.py
python figures/scripts/merge_panels.py
python src/b3_distribution_mismatch.py
```

### 5. Verify numerical claims

Three independent verification layers:

```bash
python tools/check_consistency.py     # text ↔ numbers.md
python tools/verify_numbers.py        # numbers.md ↔ actual data
python tools/verify_code_truth.py     # manuscript prose ↔ code
```

All three must report no discrepancies.

## Data Provenance

The original raw aerial radiation trajectory data were collected by Kim et al. (2019)
using the Korea Institute of Nuclear Safety (KINS) airborne monitoring system over
the Fukushima Ukedo basin. The processed CPS trajectory (2,213 georeferenced points)
in `data/processed/` is shared here with permission from the original data collector.

Access to the raw measurement records, including detector calibration data and
unprocessed gamma-ray spectra, is governed by KINS institutional data-sharing
arrangements and can be requested from the corresponding author.

## Baseline Implementation Details

For full reproducibility:

- **IDW**: pure PyTorch implementation, power exponent p = 2.0, all observed grid cells as donors (no neighbour count restriction). Deterministic given a partition.
- **Ordinary Kriging**: PyKrige 1.7.3 `OrdinaryKriging` class with exponential variogram model, isotropic range 22.5 grid cells (≈ 225 m at the 10 m × 10 m grid resolution), nugget 0.0, sill fitted automatically per split from the 50% input data variance.
- **Physics-aware U-Net**: encoder-decoder with group normalisation (8 groups; chosen over batch normalisation for stability at batch size 16), ReLU activations, five-channel input (sparse aerial, IDW baseline, land-water scalar prior, measurement mask, water mask), composite SmoothL1 loss (L = L_surface + 2.0 × L_aerial), trained 30 epochs with Adam at lr=1e-3, batch size 16, on 1,000 synthetic training samples + 200 validation samples.

## Citation

```bibtex
@article{kim2026ukedo,
  author  = {Kim, Byoung-Jik},
  title   = {Physics-aware deep learning reconstructs ground contamination from
             sparse UAV radiation measurements over the Fukushima Ukedo basin
             without field training},
  journal = {Remote Sensing},
  year    = {2026},
  note    = {in review}
}
```

## License

- **Code**: MIT License (see [LICENSE](LICENSE))
- **Data**: CC BY 4.0 (attribution required for the processed Ukedo trajectory)

## Disclosure

The author is the founder and Chief Executive Officer of Reversible Inc., which
develops commercial radiation mapping software (RadScopeX) based in part on the
methodology described in the associated manuscript and implemented in this
repository. The research design, data analysis, and conclusions reported in the
manuscript were not influenced by this commercial interest.

## Contact

Byoung-Jik Kim
Department of Nuclear Engineering, Changwon National University, Changwon 51140, Republic of Korea
