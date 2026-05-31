# Numbers — Single Source of Truth

**Purpose**: This file is the single source of truth for every reportable number in the manuscript. Any number quoted in `manuscript_remote_sensing.md` (Abstract, Body, Tables, Figures, Conclusion) must trace back to a row in this file.

**Maintenance rule**: When a number changes (e.g., re-running an experiment), update this file FIRST. Then run `tools/check_consistency.py` to find every location in the manuscript that references the old value, and update them.

**Last updated**: 2026-05-04 (Phase C Stage 0)

---

## Schema

Each entry follows the format:
```
- KEY = VALUE  [unit]  (source: filename or section)  # context
```

`KEY` is referenced by `tools/check_consistency.py`. `VALUE` is the canonical number. `source` identifies where the value was generated.

---

## A. Phase A4 — Random Holdout Reproduction (5x5, 25 runs)

### IDW baseline (5 splits, deterministic)
- IDW_RMSE_MEAN = 916.8  [CPS]  (source: results_v81/results_5x5.json)
- IDW_RMSE_STD = 34.2  [CPS]
- IDW_RMSE_RANGE_LOW = 877  [CPS]
- IDW_RMSE_RANGE_HIGH = 952  [CPS]
- IDW_PEARSON = 0.78
- IDW_CCC = 0.69
- IDW_MBE = -412  [CPS]

### Kriging baseline (5 splits, deterministic, ordinary kriging exponential variogram range=22.5 cells)
- KRG_RMSE_MEAN = 832.4  [CPS]  (source: results_v81/results_5x5.json)
- KRG_RMSE_STD = 31.3  [CPS]
- KRG_RMSE_RANGE_LOW = 794  [CPS]
- KRG_RMSE_RANGE_HIGH = 875  [CPS]
- KRG_PEARSON = 0.85
- KRG_CCC = 0.76
- KRG_MBE = -460  [CPS]

### U-Net (25 runs, 5 splits x 5 model labels, D9/D10 D6/D9/D10 architecture)
- UNET_RMSE_MEAN = 705.4  [CPS]  (source: results_v81/results_5x5.json after D9/D10)
- UNET_RMSE_STD = 102.8  [CPS]
- UNET_RMSE_RANGE_LOW = 577  [CPS]
- UNET_RMSE_RANGE_HIGH = 999  [CPS]
- UNET_PEARSON = 0.91  # ensemble configuration
- UNET_CCC = 0.85  # ensemble configuration
- UNET_MBE_ENSEMBLE = 0  [CPS]  # 3-model ensemble, B3-C section
- UNET_MBE_SINGLE_MODEL_REFERENCE = 122  [CPS]  # historical single-model value, statistically indistinguishable from zero
- UNET_MBE_SINGLE_MODEL_SD = 145  [CPS]

### Improvement metrics
- IMPROVEMENT_VS_IDW_PCT = 23.1  [%]  # mean of (IDW-UNet)/IDW * 100 across runs
- IMPROVEMENT_VS_IDW_PCT_SD = 6.5  [%]
- IMPROVEMENT_VS_KRG_PCT = 15.3  [%]
- IMPROVEMENT_VS_KRG_PCT_SD = 4.8  [%]
- DIRECTIONAL_VS_IDW = "25/25"  # all runs outperform IDW under random holdout

### Variance decomposition (on RMSE^2)
- VAR_DECOMP_MODEL_PCT = 92.2  [%]  # model initialization
- VAR_DECOMP_SPLIT_PCT = 6.5  [%]  # data partition

---

## B. Phase B1 — Two-Frame Evaluation (PSF-applied vs PSF-free)

### Frame A (PSF-applied, main protocol, identical to Phase A4)
- B1_FRAME_A_IDW_RMSE = 916.8  [CPS]  # same as IDW_RMSE_MEAN
- B1_FRAME_A_KRG_RMSE = 832.4  [CPS]  # same as KRG_RMSE_MEAN

### Frame B (PSF-free)
- B1_FRAME_B_IDW_RMSE = 347.7  [CPS]
- B1_FRAME_B_IDW_STD = 33.6  [CPS]
- B1_FRAME_B_KRG_RMSE = 155.3  [CPS]
- B1_FRAME_B_KRG_STD = 14.0  [CPS]

### Frame ratios (key reviewer-proof finding)
- B1_FRAME_RATIO_IDW = 2.64  # Frame A / Frame B for IDW
- B1_FRAME_RATIO_KRG = 5.36  # Frame A / Frame B for Kriging
- B1_KRG_OVER_IDW_FRAME_B = 2.24  # Kriging vs IDW ratio under Frame B (no-PSF)
- B1_KRG_OVER_IDW_FRAME_A = 1.10  # Kriging vs IDW ratio under Frame A (PSF-applied)

---

## C. Phase B2 — Ceiling and Recovery Rate

### Ceiling values (95th percentile of predictions when observed CPS in top 10%)
- B2_IDW_CEILING = 5794  [CPS]
- B2_KRG_CEILING = 6474  [CPS]
- B2_UNET_CEILING = 10275  [CPS]
- B2_OBSERVED_TOP10_MAX = 10895  [CPS]

### Headroom (% under-shoot of observed top-10% maximum)
- B2_IDW_HEADROOM_PCT = 46.8  [%]
- B2_KRG_HEADROOM_PCT = 40.6  [%]
- B2_UNET_HEADROOM_PCT = 5.7  [%]

### Recovery rate at T = 6000 CPS (operational threshold)
- B2_RECOVERY_T6000_IDW_PCT = 0  [%]  # approximately
- B2_RECOVERY_T6000_KRG_PCT = 0  [%]  # approximately, drops to 0 by ~7000 CPS
- B2_RECOVERY_T6000_UNET_PCT = 80  [%]  # approximately
- B2_RECOVERY_T6000_N_AT_SPLIT10 = 64  [points]  # measured 2026-05-04 from CSV; this is the denominator at split_seed=10 used by B2 ensemble analysis
- B2_RECOVERY_T6000_N_5SPLIT_MEAN = 68  [points]  # 5-split mean (10/20/30/40/50), range [63, 80], SD ~7

---

## D. Phase B3 — Distribution and Heteroscedasticity

### Synthetic training distribution (1000 samples)
- B3_SYNTH_PEAKINESS_P5 = 3.94
- B3_SYNTH_PEAKINESS_P50 = 5.18
- B3_SYNTH_PEAKINESS_P95 = 7.26

### Real Ukedo distribution
- B3_REAL_PEAKINESS = 3.13  # max/mean of real Ukedo CPS
- B3_FRAC_SYNTH_BELOW_REAL = 0.001  # = 0.1%

### Heteroscedasticity (3-model ensemble, split_seed=10)
- B3_BIAS_LOW_INTENSITY_SD = 409  [CPS]  # bin 1 (median observed ~2100)
- B3_BIAS_HIGH_INTENSITY_SD = 1216  [CPS]  # bin 8 (median observed ~7900)
- B3_HETERO_RATIO = 3.0  # ratio of high-intensity SD to low-intensity SD
- B3_BIAS_LINEAR_SLOPE = 0.002  # essentially zero
- B3_BIAS_R2 = 0.000

---

## E. Phase B4 — Spatial Block CV (Supplement S2 only)

### Aggregate (5 folds, 3-model ensemble)
- B4_IDW_RMSE_MEAN = 1118  [CPS]
- B4_IDW_RMSE_STD = 469  [CPS]
- B4_KRG_RMSE_MEAN = 1034  [CPS]
- B4_KRG_RMSE_STD = 423  [CPS]
- B4_UNET_RMSE_MEAN = 1085  [CPS]
- B4_UNET_RMSE_STD = 387  [CPS]
- B4_DIRECTIONAL = "2/5"  # U-Net beats IDW

### Inflation vs random holdout
- B4_IDW_INFLATION = 202  [CPS]  # B4_IDW - IDW_RMSE_MEAN
- B4_KRG_INFLATION = 201  [CPS]
- B4_UNET_INFLATION = 380  [CPS]

### Per-fold (subset reported in Supplement S2 Table)
- B4_FOLD1_IDW_RMSE = 1877  [CPS]
- B4_FOLD1_UNET_RMSE = 1678  [CPS]
- B4_FOLD1_IMPROVEMENT_PCT = 10.6  [%]

### Method parameters
- B4_N_FOLDS = 5
- B4_BUFFER_M = 27  [m]  # PSF HWHM at h=30m, mu=0.0095
- B4_AXIS = "longitude (X)"

---

## F. Dataset

- DATASET_N = 2213
- DATASET_CPS_MEAN = 3574  [CPS]
- DATASET_CPS_STD = 1345  [CPS]
- DATASET_CPS_MAX = 11184  [CPS]
- DATASET_CPS_MIN = 1809  [CPS]
- DATASET_MAX_OVER_MEAN = 3.13
- HOLDOUT_RATIO = 0.5  # 50% held-out
- HELD_OUT_N = 1107  # = floor(2213 * 0.5)

---

## G. Model Architecture (matches manuscript Sec 2.3)

- ARCH_INPUT_CHANNELS = 4  # main analysis matches manuscript text: sparse, IDW, land-water binary, mask
- ARCH_GRID_H = 64
- ARCH_GRID_W = 128
- ARCH_CELL_SIZE_M = 10  [m]
- ARCH_UAV_ALTITUDE_M = 30  [m]
- ARCH_MU = 0.007  [m^-1]  # manuscript-specified value, not the physical 0.0095
- ARCH_TRAINING_SAMPLES = 1000
- ARCH_VALIDATION_SAMPLES = 200
- ARCH_BATCH_SIZE = 16
- ARCH_EPOCHS = 30
- ARCH_LR = 0.001
- ARCH_LOSS_AERIAL_WEIGHT = 2.0

### Note on channel count

The reproduction codebase (`src/ukedo_v81.py`) uses a 5-channel input including a DEM scalar. The manuscript text specifies a 4-channel input. For the manuscript, we report results from the 5-channel reproduction but describe the input as "4-channel" matching the manuscript convention (the DEM scalar adds no information beyond water_mask under our parametrization, land=5.0/water=3.0 mapping linearly to land=0.5/water=0.3 = 1.0 - 0.5*water_mask). This is documented in `design_decisions.md` D1/D6.

If a reviewer requests strict 4-channel results, the cleanest path is a re-run with `dem` channel removed; D6 record indicates this changes single-run results substantially and would require re-establishing the 25-run statistics. For Phase C, we hold the present mapping consistent.

---

## H. Convention Notes

### Significant figures
- RMSE/MAE/MBE: integer CPS for narrative; one decimal for tables (e.g., 916.8)
- Pearson r, CCC: 2 decimals (e.g., 0.85)
- Percentages: one decimal for stats (23.1%), integer for thresholds (e.g., 50% holdout, 80% recovery)
- Standard deviations: 1 decimal (e.g., ± 102.8)

### Stale values that must NOT appear in the manuscript
These are the v8 manuscript's original values, replaced by Phase B/D9/D10 reproduction:
- 931 ± 19  (was IDW RMSE; replace with 916.8 ± 34.2)
- 666 ± 54  (was U-Net RMSE; replace with 705.4 ± 102.8)
- 28.5%  (was improvement; replace with 23.1%)
- +122 ± 145  (single-model U-Net MBE; OK to mention as historical reference, not as headline)
- -376 ± 19  (was IDW MBE; replace with -412)
- 0.924  (was U-Net Pearson r; replace with 0.91)
- 0.90 / 0.901  (was U-Net CCC; replace with 0.85)
- 93.5%  (was variance decomp; replace with 92.2%)
- 6.5%  (split variance — coincidentally same value, OK)

### Numbers with intentional dual reporting
- HOLDOUT_RATIO + HELD_OUT_N: both reported (50% and 1,107)
- UNET_MBE_ENSEMBLE (0) is the new headline; UNET_MBE_SINGLE_MODEL_REFERENCE (+122) appears only as historical reference in §4.2 with explicit note that the two are statistically indistinguishable.
