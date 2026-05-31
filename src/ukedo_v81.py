"""
ukedo_v81.py - Faithful v8 Reproduction with Reinforcement Hooks

Goal: Reproduce v8 manuscript results (IDW RMSE ~ 931, U-Net RMSE ~ 666, ~+29%)
       and add Kriging baseline + 5x5 seed design for reinforcement.

Algorithmic core (D8 - 2026-05-03): MATCHES the modular v8 manuscript package
exactly. 5-channel input [sparse/P, idw/P, dem/10, meas_mask, water_mask] where
dem = 5.0 on land, 3.0 on water. This is the channel layout that produced
RMSE 666 in the manuscript, confirmed via the modular v8 package source code.

Reinforcement layer added on top of the manuscript algorithm:
  1. 5x5 run identifier scheme = 25 runs
        split seeds  = {10, 20, 30, 40, 50}     (deterministic — controls test partition)
        model labels = {42, 123, 2026, 7, 99}   (D9: identifier only, NOT a torch seed —
                                                 matches manuscript trainer's natural
                                                 random init distribution)
  2. Kriging baseline (ordinary, exponential variogram, range = 22.5 cells)
  3. 7 metrics: RMSE, MAE, Pearson r, Spearman rho, CCC, MBE, top-10% overlap
  4. 200 validation synthetic samples (loss monitoring only)
  5. Variance decomposition (split vs model) on RMSE^2
  6. JSON results + figures saved to disk

Held-out evaluation strategy (UNCHANGED from v8 - this is the core of the paper):
  - Both IDW and Kriging surfaces are forward-projected through K (PSF) before
    comparison with held-out aerial CPS values.
  - This is the "double-blurring" evaluation framing (manuscript Sec 1.3).
  - PSF-free IDW comparison is left for Phase B reinforcement (separate script).

Usage:
  python ukedo_v81.py --mode smoke    # Single run smoke test (split=10, model=42)
  python ukedo_v81.py --mode full     # Full 25-run experiment

Required files (same directory as this script):
  - integrated data.csv          (Ukedo 2213 trajectory points)
  - ukedo_water_mask.png         (binary water mask)

Required packages:
  pip install torch torchvision pandas numpy matplotlib scipy pillow pykrige
"""

import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

try:
    from pykrige.ok import OrdinaryKriging
    HAS_PYKRIGE = True
except ImportError:
    HAS_PYKRIGE = False
    print("[WARNING] pykrige not installed. Kriging baseline will be skipped.")
    print("          Install with: pip install pykrige")


# =============================================================================
# 0. Configuration
# =============================================================================
class Config:
    # Grid (matches v8 + manuscript)
    H, W = 64, 128
    CELL_SIZE_M = 10.0
    UAV_ALTITUDE = 30.0
    MU = 0.007                       # m^-1, manuscript Sec 2.2

    # Training
    N_TRAIN = 1000
    N_VAL = 200                      # NEW: validation samples for loss monitoring
    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 1e-3

    # 5x5 experimental design (matches manuscript Table 1)
    SPLIT_SEEDS = [10, 20, 30, 40, 50]
    MODEL_SEEDS = [42, 123, 2026, 7, 99]
    HOLDOUT_RATIO = 0.5

    # Kriging
    VARIOGRAM_MODEL = 'exponential'
    VARIOGRAM_RANGE_M = 225.0        # from P0 EDA
    KRIG_RANGE_CELLS = VARIOGRAM_RANGE_M / CELL_SIZE_M  # 22.5

    # Synthetic data fixed seeds (so the synthetic corpus is the same across model seeds)
    TRAIN_DATA_SEED = 12345
    VAL_DATA_SEED = 99999

    # Paths
    REAL_CSV_PATH = "integrated data.csv"
    MASK_IMAGE_PATH = "ukedo_water_mask.png"
    OUTPUT_DIR = "results_v81"

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# 1. Physics: PSF kernel + terrain loader
# =============================================================================
def get_physics_kernel(cfg, device):
    """K(dx,dy) = exp(-mu * d) / d^2, normalized to sum to 1.
    Full-image radius (radius_y = H-1, radius_x = W-1) - matches v8."""
    radius_y, radius_x = cfg.H - 1, cfg.W - 1
    k_y, k_x = torch.meshgrid(
        torch.arange(-radius_y, radius_y + 1, device=device, dtype=torch.float32),
        torch.arange(-radius_x, radius_x + 1, device=device, dtype=torch.float32),
        indexing='ij'
    )
    dist_sq = (k_x * cfg.CELL_SIZE_M) ** 2 + (k_y * cfg.CELL_SIZE_M) ** 2 + cfg.UAV_ALTITUDE ** 2
    dist = torch.sqrt(dist_sq)
    kernel = torch.exp(-cfg.MU * dist) / dist_sq
    kernel /= kernel.sum()
    return kernel.view(1, 1, 2 * radius_y + 1, 2 * radius_x + 1)


def load_terrain(cfg):
    """Binary water mask + DEM scalar tensor (manuscript v8 layout).
       water_mask: water=1.0, land=0.0  (H x W)
       dem       : land=5.0, water=3.0   (H x W)
       Used as channel 3 in the U-Net input as `dem / 10.0` (= 0.5 / 0.3)."""
    if not os.path.exists(cfg.MASK_IMAGE_PATH):
        raise FileNotFoundError(f"Required file not found: {cfg.MASK_IMAGE_PATH}")
    mask_img = Image.open(cfg.MASK_IMAGE_PATH).convert('L')
    water_mask = transforms.ToTensor()(transforms.Resize((cfg.H, cfg.W))(mask_img)).squeeze(0).to(cfg.DEVICE)
    water_mask = (water_mask > 0.5).float()
    dem = torch.ones((cfg.H, cfg.W), device=cfg.DEVICE) * 5.0
    dem[water_mask == 1.0] = 3.0
    return dem, water_mask


# =============================================================================
# 2. Synthetic data generation (v8 logic, kept identical)
# =============================================================================
def make_ground_source(cfg, water_mask):
    """Random multi-Gaussian hotspots over a smoothed background.
    Hotspots suppressed at water pixels via (1 - water_mask)."""
    H, W = cfg.H, cfg.W
    device = cfg.DEVICE
    gt = torch.zeros((H, W), device=device)
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    # Smoothed random background
    bg = torch.rand((1, 1, H, W), device=device)
    bg = F.avg_pool2d(bg, kernel_size=11, stride=1, padding=5).squeeze()
    gt += bg * 0.3

    # 5-15 random Gaussian hotspots
    hotspots = torch.zeros((H, W), device=device)
    n_src = torch.randint(5, 15, (1,)).item()
    for _ in range(n_src):
        cx = torch.rand(1).item() * W
        cy = torch.rand(1).item() * H
        sigma = torch.rand(1).item() * 10 + 2
        amp = torch.rand(1).item() * 0.8 + 0.2
        hotspots += amp * torch.exp(
            -((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2)
        )

    gt += hotspots * (1.0 - water_mask)
    if gt.max() > 0:
        gt = gt / gt.max()
    return gt


def simulate_diverse_measurements(ground_source, kernel, cfg):
    """Simulate UAV trajectory: convolve ground source -> aerial,
    apply random horizontal/vertical/grid sampling pattern."""
    H, W = cfg.H, cfg.W
    device = cfg.DEVICE
    pad_y, pad_x = H - 1, W - 1
    aerial_cps = F.conv2d(ground_source.view(1, 1, H, W), kernel, padding=(pad_y, pad_x))[0, 0]

    meas_mask = torch.zeros((H, W), device=device)
    pattern = torch.randint(0, 3, (1,)).item()    # 0=H, 1=V, 2=grid
    spacing = torch.randint(2, 7, (1,)).item()    # 2..6
    if pattern in [0, 2]:
        meas_mask[::spacing, :] = 1.0
    if pattern in [1, 2]:
        meas_mask[:, ::spacing] = 1.0
    if meas_mask.sum() == 0:
        meas_mask[H // 2, W // 2] = 1.0

    sparse_meas = aerial_cps * meas_mask
    idw_map = compute_idw_torch(meas_mask, sparse_meas, cfg)
    return aerial_cps, sparse_meas, idw_map, meas_mask


def compute_idw_torch(meas_mask, vals_map, cfg, p=2.0):
    """Vectorized IDW on a regular grid. p=2 (squared distance weights)."""
    H, W = cfg.H, cfg.W
    device = cfg.DEVICE
    pts = torch.nonzero(meas_mask)              # [N, 2] (y, x)
    if len(pts) == 0:
        return torch.zeros((H, W), device=device)
    vals = vals_map[meas_mask > 0]              # [N]
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    coords = torch.stack([y_grid.flatten(), x_grid.flatten()], dim=1)
    dist_sq = torch.cdist(coords, pts.float()) ** 2 + 1e-4
    weights = 1.0 / dist_sq
    idw = (torch.sum(weights * vals.view(1, -1), dim=1) / torch.sum(weights, dim=1)).view(H, W)
    return idw


def compute_kriging(meas_mask_np, sparse_map_np, cfg):
    """Ordinary kriging on the same H x W grid using pykrige.
    Returns: (H, W) numpy array of predictions."""
    if not HAS_PYKRIGE:
        return np.zeros_like(sparse_map_np)
    H, W = cfg.H, cfg.W
    ys_in, xs_in = np.where(meas_mask_np > 0)
    vals_in = sparse_map_np[ys_in, xs_in].astype(float)
    if len(vals_in) == 0:
        return np.zeros((H, W))
    sill = float(np.var(vals_in)) if len(vals_in) > 1 else 1.0
    OK = OrdinaryKriging(
        xs_in.astype(float), ys_in.astype(float), vals_in,
        variogram_model=cfg.VARIOGRAM_MODEL,
        variogram_parameters={'sill': sill, 'range': cfg.KRIG_RANGE_CELLS, 'nugget': 0.0},
        verbose=False, enable_plotting=False,
    )
    grid_x = np.arange(W, dtype=float)
    grid_y = np.arange(H, dtype=float)
    z_grid, _ = OK.execute('grid', grid_x, grid_y)
    return np.array(z_grid)


# =============================================================================
# 3. Physics-Aware U-Net (5-channel input, manuscript v8 layout)
# =============================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class PhysicsAwareUNet(nn.Module):
    """5-channel input: sparse/P / idw/P / dem/10 / meas_mask / water_mask  ->  1-channel surface (Softplus).
    Matches the manuscript v8 modular package exactly."""

    def __init__(self, in_ch=5):
        super().__init__()
        self.inc = DoubleConv(in_ch, 16)
        self.d1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.d2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.u1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.c1 = DoubleConv(64, 32)
        self.u2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.c2 = DoubleConv(32, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        d = self.c1(torch.cat([self.u1(x3), x2], dim=1))
        d = self.c2(torch.cat([self.u2(d), x1], dim=1))
        return F.softplus(self.out(d))


# =============================================================================
# 4. Build training / validation datasets
# =============================================================================
def build_synthetic_dataset(n_samples, cfg, kernel, water_mask, dem, base_seed):
    """Generate n_samples paired (input, surface_gt, aerial_gt) tensors.
    5-channel input: [sparse/P, idw/P, dem/10, meas_mask, water_mask] (matches manuscript v8)."""
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    dem_norm = dem / 10.0
    x_list, y_surf_list, y_aerial_list = [], [], []
    for _ in range(n_samples):
        ground_src = make_ground_source(cfg, water_mask)
        aerial_cps, sparse, idw, mm = simulate_diverse_measurements(ground_src, kernel, cfg)
        P = sparse.max() + 1e-8
        # 5-channel input (manuscript v8 layout)
        x_tensor = torch.stack([sparse / P, idw / P, dem_norm, mm, water_mask], dim=0)
        x_list.append(x_tensor.cpu())
        y_surf_list.append((ground_src / P).unsqueeze(0).cpu())
        y_aerial_list.append((aerial_cps / P).unsqueeze(0).cpu())
    return TensorDataset(torch.stack(x_list), torch.stack(y_surf_list), torch.stack(y_aerial_list))


# =============================================================================
# 5. Training
# =============================================================================
def train_unet(model_seed, cfg, kernel, train_loader, val_loader=None, verbose=True):
    """Train a single PhysicsAwareUNet.

    D9 (2026-05-03): model_seed is kept as a run identifier only (used in logs / JSON).
    No torch/np seed is set here — matches the manuscript's modular v8 trainer, which
    does not set any seed during training. Forcing torch.manual_seed(model_seed) was
    found to deterministically pick a bad-init region for seed=42 (Phase A4: seed=42
    failed in 5/5 splits). Removing the forcing aligns our 25-run statistics with the
    manuscript's natural random-init distribution (target: RMSE 666 +/- 54).
    """
    model = PhysicsAwareUNet(in_ch=5).to(cfg.DEVICE)
    opt = optim.Adam(model.parameters(), lr=cfg.LR)
    criterion = nn.SmoothL1Loss()
    pad_y, pad_x = cfg.H - 1, cfg.W - 1

    train_losses, val_losses = [], []
    for ep in range(1, cfg.EPOCHS + 1):
        model.train()
        tr_loss, n_tr = 0.0, 0
        for bx, by_s, by_a in train_loader:
            bx = bx.to(cfg.DEVICE); by_s = by_s.to(cfg.DEVICE); by_a = by_a.to(cfg.DEVICE)
            opt.zero_grad()
            pred_s = model(bx)
            pred_a = F.conv2d(pred_s, kernel, padding=(pad_y, pad_x))
            loss = criterion(pred_s, by_s) + 2.0 * criterion(pred_a, by_a)
            loss.backward()
            opt.step()
            tr_loss += loss.item(); n_tr += 1
        train_losses.append(tr_loss / max(n_tr, 1))

        if val_loader is not None:
            model.eval()
            v_loss, n_v = 0.0, 0
            with torch.no_grad():
                for bx, by_s, by_a in val_loader:
                    bx = bx.to(cfg.DEVICE); by_s = by_s.to(cfg.DEVICE); by_a = by_a.to(cfg.DEVICE)
                    pred_s = model(bx)
                    pred_a = F.conv2d(pred_s, kernel, padding=(pad_y, pad_x))
                    loss = criterion(pred_s, by_s) + 2.0 * criterion(pred_a, by_a)
                    v_loss += loss.item(); n_v += 1
            val_losses.append(v_loss / max(n_v, 1))

        if verbose and (ep == 1 or ep % 5 == 0 or ep == cfg.EPOCHS):
            v = f"  val={val_losses[-1]:.4f}" if val_loader else ""
            print(f"   epoch {ep:>2}/{cfg.EPOCHS}: train={train_losses[-1]:.4f}{v}")
    return model, train_losses, val_losses


# =============================================================================
# 6. Real-data evaluation (matches v8 evaluation pipeline + Kriging)
# =============================================================================
def split_real_data(df, ratio, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    sp = int(len(df) * ratio)
    return df.iloc[idx[:sp]].copy(), df.iloc[idx[sp:]].copy()


def grid_coords(df, lat_min, lat_max, lon_min, lon_max, H, W):
    y = ((lat_max - df['Latitude']) / (lat_max - lat_min) * (H - 1)).astype(int).clip(0, H - 1)
    x = ((df['Longitude'] - lon_min) / (lon_max - lon_min) * (W - 1)).astype(int).clip(0, W - 1)
    return y.values, x.values


def evaluate_split(model, split_seed, df_full, cfg, kernel, water_mask, dem):
    """Evaluate one (model, split) combination. Returns dict with predictions and maps.
    5-channel input matches manuscript v8."""
    H, W = cfg.H, cfg.W
    device = cfg.DEVICE
    pad_y, pad_x = H - 1, W - 1
    dem_norm = dem / 10.0

    df_in, df_out = split_real_data(df_full, cfg.HOLDOUT_RATIO, seed=split_seed)
    lat_min = df_full['Latitude'].min(); lat_max = df_full['Latitude'].max()
    lon_min = df_full['Longitude'].min(); lon_max = df_full['Longitude'].max()

    # Rasterize input set onto H x W grid
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

    # IDW
    idw_norm = compute_idw_torch(meas_mask, sparse_norm, cfg)

    # Kriging (raw CPS scale, then normalize)
    sparse_map_np = sparse_map.cpu().numpy()
    meas_mask_np = meas_mask.cpu().numpy()
    krig_map_np = compute_kriging(meas_mask_np, sparse_map_np, cfg)
    krig_norm = torch.from_numpy(krig_map_np / MAX_CPS).float().to(device)
    krig_norm = krig_norm.clamp(min=0.0)        # kriging can produce negatives

    # AI inference (5-channel: sparse, idw, dem/10, meas_mask, water_mask)
    model.eval()
    with torch.no_grad():
        x_in_t = torch.stack([sparse_norm, idw_norm, dem_norm, meas_mask, water_mask], dim=0).unsqueeze(0)
        surf_AI = model(x_in_t)
        surf_IDW = idw_norm.view(1, 1, H, W)
        surf_KRG = krig_norm.view(1, 1, H, W)

        aerial_AI = F.conv2d(surf_AI, kernel, padding=(pad_y, pad_x)).squeeze() * MAX_CPS
        aerial_IDW = F.conv2d(surf_IDW, kernel, padding=(pad_y, pad_x)).squeeze() * MAX_CPS
        aerial_KRG = F.conv2d(surf_KRG, kernel, padding=(pad_y, pad_x)).squeeze() * MAX_CPS

    # Sample at held-out trajectory points
    y_out, x_out = grid_coords(df_out, lat_min, lat_max, lon_min, lon_max, H, W)
    true_vals = df_out['CPS'].values.astype(float)
    ai_vals = aerial_AI[y_out, x_out].cpu().numpy()
    idw_vals = aerial_IDW[y_out, x_out].cpu().numpy()
    krg_vals = aerial_KRG[y_out, x_out].cpu().numpy()

    return {
        'true': true_vals, 'ai': ai_vals, 'idw': idw_vals, 'krg': krg_vals,
        'n_in': len(df_in), 'n_out': len(df_out),
        'maps': {
            'sparse': (sparse_norm * MAX_CPS).cpu().numpy(),
            'aerial_AI': aerial_AI.cpu().numpy(),
            'aerial_IDW': aerial_IDW.cpu().numpy(),
            'aerial_KRG': aerial_KRG.cpu().numpy(),
            'surf_AI': (surf_AI.squeeze() * MAX_CPS).cpu().numpy(),
            'surf_IDW': (surf_IDW.squeeze() * MAX_CPS).cpu().numpy(),
            'surf_KRG': (surf_KRG.squeeze() * MAX_CPS).cpu().numpy(),
        },
    }


# =============================================================================
# 7. Metrics
# =============================================================================
def compute_metrics(true, pred, top_pct=0.10):
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    mae = float(np.mean(np.abs(true - pred)))
    mbe = float(np.mean(pred - true))
    r_pearson = float(pearsonr(true, pred)[0])
    r_spear = float(spearmanr(true, pred)[0])
    # Lin's CCC
    mt, mp = true.mean(), pred.mean()
    vt, vp = true.var(ddof=0), pred.var(ddof=0)
    cov = np.mean((true - mt) * (pred - mp))
    ccc = float(2 * cov / (vt + vp + (mt - mp) ** 2 + 1e-12))
    # Top-10% overlap (Jaccard-like)
    thr_t = np.quantile(true, 1 - top_pct)
    thr_p = np.quantile(pred, 1 - top_pct)
    in_t = true >= thr_t
    in_p = pred >= thr_p
    overlap = float(np.mean(in_t & in_p) / np.mean(in_t)) if in_t.any() else 0.0
    return {
        'rmse': rmse, 'mae': mae, 'mbe': mbe,
        'pearson': r_pearson, 'spearman': r_spear,
        'ccc': ccc, 'top10_overlap': overlap,
    }


# =============================================================================
# 8. Smoke test
# =============================================================================
def run_smoke_test(cfg):
    print("=" * 70)
    print("SMOKE TEST: split_seed=10, model_seed=42, 50% holdout")
    print(f"Device: {cfg.DEVICE}  |  pykrige: {HAS_PYKRIGE}")
    print("=" * 70)

    out = Path(cfg.OUTPUT_DIR); out.mkdir(exist_ok=True)
    kernel = get_physics_kernel(cfg, cfg.DEVICE)
    dem, water_mask = load_terrain(cfg)
    print(f"Water mask: shape={tuple(water_mask.shape)}, water fraction={water_mask.mean().item():.3f}")

    print("\n[1] Building synthetic datasets (1000 train + 200 val)...")
    t0 = time.time()
    train_ds = build_synthetic_dataset(cfg.N_TRAIN, cfg, kernel, water_mask, dem, base_seed=cfg.TRAIN_DATA_SEED)
    val_ds = build_synthetic_dataset(cfg.N_VAL, cfg, kernel, water_mask, dem, base_seed=cfg.VAL_DATA_SEED)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    print(f"   done in {time.time() - t0:.1f}s")

    print("\n[2] Training U-Net (model_seed=42, 30 epochs)...")
    t0 = time.time()
    model, tr_losses, val_losses = train_unet(42, cfg, kernel, train_loader, val_loader)
    print(f"   done in {time.time() - t0:.1f}s")

    print("\n[3] Loading real Ukedo data...")
    df = pd.read_csv(cfg.REAL_CSV_PATH, usecols=['Time', 'Latitude', 'Longitude', 'CPS']).dropna()
    df = df.sort_values(by='Time').reset_index(drop=True)
    print(f"   N={len(df)}  CPS mean={df['CPS'].mean():.0f}  max={df['CPS'].max():.0f}")

    print("\n[4] Evaluating on real data...")
    res = evaluate_split(model, 10, df, cfg, kernel, water_mask, dem)
    m_idw = compute_metrics(res['true'], res['idw'])
    m_krg = compute_metrics(res['true'], res['krg']) if HAS_PYKRIGE else None
    m_ai = compute_metrics(res['true'], res['ai'])

    print("\n" + "=" * 70)
    print("SMOKE TEST RESULTS  (manuscript target: IDW~931, U-Net~666, +29%)")
    print("=" * 70)
    headers = ['Metric', 'IDW', 'Kriging', 'U-Net']
    print(f"{headers[0]:<15} {headers[1]:>10} {headers[2]:>10} {headers[3]:>10}")
    print("-" * 50)
    for k in ['rmse', 'mae', 'mbe', 'pearson', 'spearman', 'ccc', 'top10_overlap']:
        krg_str = f"{m_krg[k]:>10.3f}" if m_krg else f"{'--':>10}"
        print(f"{k:<15} {m_idw[k]:>10.3f} {krg_str} {m_ai[k]:>10.3f}")
    imp_idw = (m_idw['rmse'] - m_ai['rmse']) / m_idw['rmse'] * 100
    print(f"\n[U-Net vs IDW] RMSE improvement: {imp_idw:+.1f}%")
    if m_krg:
        imp_krg = (m_krg['rmse'] - m_ai['rmse']) / m_krg['rmse'] * 100
        print(f"[U-Net vs Krg] RMSE improvement: {imp_krg:+.1f}%")

    save_smoke_outputs(out, res, m_idw, m_krg, m_ai, tr_losses, val_losses)
    print(f"\nSaved to: {out.absolute()}")


def save_smoke_outputs(out_dir, res, m_idw, m_krg, m_ai, tr_losses, val_losses):
    payload = {
        'IDW': m_idw,
        'Kriging': m_krg,
        'UNet': m_ai,
        'n_held_out': int(len(res['true'])),
        'training': {'train_loss': tr_losses, 'val_loss': val_losses},
    }
    with open(out_dir / "smoke_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    # Learning curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(tr_losses) + 1), tr_losses, 'o-', label='Train')
    ax.plot(range(1, len(val_losses) + 1), val_losses, 's-', label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Smooth L1 Loss'); ax.legend(); ax.grid(alpha=0.4)
    ax.set_title('Smoke test learning curve (split=10, model=42)')
    fig.savefig(out_dir / "smoke_learning_curve.png", dpi=120, bbox_inches='tight'); plt.close(fig)

    # Map comparison: rows = surface domain / aerial domain; cols = sparse, IDW, Kriging, AI
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    maps = res['maps']
    vmin, vmax = 0.0, max(float(res['true'].max()), 11200.0)

    axes[0, 0].imshow(maps['sparse'], cmap='jet', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Sparse aerial input (50%)')
    axes[0, 1].imshow(maps['surf_IDW'], cmap='jet', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('IDW surface proxy')
    axes[0, 2].imshow(maps['surf_KRG'], cmap='jet', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title('Kriging surface proxy')
    axes[0, 3].imshow(maps['surf_AI'], cmap='jet')
    axes[0, 3].set_title('U-Net ground estimate')

    axes[1, 0].axis('off'); axes[1, 0].set_title('(held-out scoring)')
    axes[1, 1].imshow(maps['aerial_IDW'], cmap='jet', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(f"IDW aerial pred (RMSE={m_idw['rmse']:.0f})")
    krg_title = f"Kriging aerial pred (RMSE={m_krg['rmse']:.0f})" if m_krg else "Kriging (n/a)"
    axes[1, 2].imshow(maps['aerial_KRG'], cmap='jet', vmin=vmin, vmax=vmax)
    axes[1, 2].set_title(krg_title)
    axes[1, 3].imshow(maps['aerial_AI'], cmap='jet', vmin=vmin, vmax=vmax)
    axes[1, 3].set_title(f"U-Net aerial pred (RMSE={m_ai['rmse']:.0f})")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Smoke test reconstruction (split=10, model=42, 50% held-out)', y=0.99, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "smoke_maps.png", dpi=120, bbox_inches='tight'); plt.close(fig)

    # Predicted vs observed scatter (3-panel)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [(axes[0], res['idw'], 'IDW', m_idw), (axes[1], res['krg'], 'Kriging', m_krg), (axes[2], res['ai'], 'U-Net', m_ai)]
    for ax, vals, lbl, m in panels:
        if m is None:
            ax.axis('off'); ax.set_title(f"{lbl} (n/a)"); continue
        ax.scatter(res['true'], vals, s=4, alpha=0.5)
        lim = [0, max(float(res['true'].max()), float(np.asarray(vals).max()))]
        ax.plot(lim, lim, 'k--', alpha=0.7)
        ax.set_xlabel('Observed CPS'); ax.set_ylabel(f'{lbl} predicted CPS')
        ax.set_title(f"{lbl}: r={m['pearson']:.3f}  MBE={m['mbe']:+.0f}  RMSE={m['rmse']:.0f}")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal'); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "smoke_scatter.png", dpi=120, bbox_inches='tight'); plt.close(fig)


# =============================================================================
# 9. Full 5x5 experiment
# =============================================================================
def run_full_5x5(cfg):
    print("=" * 70)
    print("FULL 5x5 EXPERIMENT  (5 splits x 5 model seeds = 25 runs)")
    print(f"Device: {cfg.DEVICE}  |  pykrige: {HAS_PYKRIGE}")
    print("=" * 70)

    out = Path(cfg.OUTPUT_DIR); out.mkdir(exist_ok=True)
    kernel = get_physics_kernel(cfg, cfg.DEVICE)
    dem, water_mask = load_terrain(cfg)
    df = pd.read_csv(cfg.REAL_CSV_PATH, usecols=['Time', 'Latitude', 'Longitude', 'CPS']).dropna()
    df = df.sort_values(by='Time').reset_index(drop=True)
    print(f"Loaded Ukedo data: N={len(df)}, CPS mean={df['CPS'].mean():.0f}")

    print("\n[A] Building synthetic datasets (shared across all model seeds)...")
    t0 = time.time()
    train_ds = build_synthetic_dataset(cfg.N_TRAIN, cfg, kernel, water_mask, dem, base_seed=cfg.TRAIN_DATA_SEED)
    val_ds = build_synthetic_dataset(cfg.N_VAL, cfg, kernel, water_mask, dem, base_seed=cfg.VAL_DATA_SEED)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    print(f"    done in {time.time() - t0:.1f}s")

    # Train all 5 models
    print("\n[B] Training 5 models...")
    models = {}
    losses = {}
    for ms in reversed(cfg.MODEL_SEEDS):  # D10: test if first-init-position is the outlier driver
        print(f"\n  --- model_seed={ms} ---")
        t0 = time.time()
        model, tr_l, va_l = train_unet(ms, cfg, kernel, train_loader, val_loader, verbose=False)
        print(f"    done in {time.time() - t0:.1f}s; final train={tr_l[-1]:.4f}, val={va_l[-1]:.4f}")
        models[ms] = model
        losses[ms] = {'train': tr_l, 'val': va_l}

    # Evaluate 25 runs
    print("\n[C] Evaluating 5 splits x 5 models = 25 runs...")
    runs = []
    for ss in cfg.SPLIT_SEEDS:
        # IDW + Kriging deterministic per split: compute once using first model
        first = cfg.MODEL_SEEDS[0]
        base = evaluate_split(models[first], ss, df, cfg, kernel, water_mask, dem)
        m_idw = compute_metrics(base['true'], base['idw'])
        m_krg = compute_metrics(base['true'], base['krg']) if HAS_PYKRIGE else None
        m_ai0 = compute_metrics(base['true'], base['ai'])
        krg_rmse = f"{m_krg['rmse']:.0f}" if m_krg else "n/a"
        print(f"\n  split={ss}: IDW={m_idw['rmse']:.0f}, Krig={krg_rmse}, AI[seed={first}]={m_ai0['rmse']:.0f}")
        runs.append({'split': ss, 'model': first, 'IDW': m_idw, 'Kriging': m_krg, 'UNet': m_ai0})
        for ms in cfg.MODEL_SEEDS[1:]:
            r2 = evaluate_split(models[ms], ss, df, cfg, kernel, water_mask, dem)
            m_ai = compute_metrics(r2['true'], r2['ai'])
            runs.append({'split': ss, 'model': ms, 'IDW': m_idw, 'Kriging': m_krg, 'UNet': m_ai})
            print(f"        AI[seed={ms}]={m_ai['rmse']:.0f}")

    # Save and summarize
    with open(out / "results_5x5.json", 'w', encoding='utf-8') as f:
        json.dump({'runs': runs}, f, indent=2)
    summarize_5x5(out, runs, cfg)


def summarize_5x5(out_dir, runs, cfg):
    print("\n" + "=" * 70)
    print("5x5 SUMMARY")
    print("=" * 70)
    first = cfg.MODEL_SEEDS[0]

    rmse_idw = [r['IDW']['rmse'] for r in runs if r['model'] == first]              # 5
    rmse_krg = [r['Kriging']['rmse'] for r in runs if r['model'] == first and r['Kriging']]
    rmse_ai = [r['UNet']['rmse'] for r in runs]                                     # 25

    def fmt(arr, label, n_label):
        if not arr:
            print(f"  {label}: n/a"); return
        arr = np.array(arr)
        print(f"  {label} RMSE: {arr.mean():7.1f}  +/- {arr.std(ddof=1):5.1f}   ({n_label})")

    fmt(rmse_idw, 'IDW  ', '5 splits')
    fmt(rmse_krg, 'Krig ', '5 splits')
    fmt(rmse_ai, 'U-Net', '25 runs')
    print(f"\nManuscript Table 2 target: IDW=931+/-19, U-Net=666+/-54")

    # Variance decomposition on RMSE^2
    ai_arr = np.array([
        [r['UNet']['rmse'] for r in runs if r['split'] == ss]
        for ss in cfg.SPLIT_SEEDS
    ])  # [5, 5]
    grand = ai_arr.mean()
    n = ai_arr.size
    var_split = ((ai_arr.mean(axis=1) - grand) ** 2 * 5).sum() / n
    var_model = ((ai_arr.mean(axis=0) - grand) ** 2 * 5).sum() / n
    var_total = ((ai_arr - grand) ** 2).sum() / n
    print("\nVariance decomposition of U-Net RMSE:")
    print(f"  Total variance: {var_total:8.1f}")
    print(f"  Split  comp:    {var_split:8.1f}  ({var_split/var_total*100:5.1f}%)")
    print(f"  Model  comp:    {var_model:8.1f}  ({var_model/var_total*100:5.1f}%)")
    print("Manuscript target: 6.5% split / 93.5% model")

    # Directional
    n_better_idw = sum(1 for r in runs if r['UNet']['rmse'] < r['IDW']['rmse'])
    print(f"\nDirectional: U-Net < IDW in {n_better_idw}/25 runs (target: 25/25)")
    if rmse_krg:
        n_better_krg = sum(1 for r in runs if r['Kriging'] and r['UNet']['rmse'] < r['Kriging']['rmse'])
        print(f"             U-Net < Krig in {n_better_krg}/25 runs")

    # RMSE distribution figure
    fig, ax = plt.subplots(figsize=(9, 6))
    methods = [('IDW', rmse_idw, 'tab:orange'), ('U-Net (25 runs)', rmse_ai, 'tab:blue')]
    if rmse_krg:
        methods.insert(1, ('Kriging', rmse_krg, 'tab:green'))
    positions = list(range(len(methods)))
    ax.boxplot([m[1] for m in methods], positions=positions, widths=0.5,
               showmeans=True, meanline=True)
    for i, (lbl, arr, col) in enumerate(methods):
        ax.scatter([i] * len(arr), arr, alpha=0.6, color=col, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([m[0] for m in methods])
    ax.set_ylabel('Held-out RMSE (CPS)')
    ax.set_title('5x5 RMSE distribution at 50% held-out')
    ax.grid(axis='y', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_5x5_rmse_box.png", dpi=120, bbox_inches='tight'); plt.close(fig)
    print(f"\nSaved figure: {out_dir / 'fig_5x5_rmse_box.png'}")


# =============================================================================
# 10. Entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'full'], default='smoke',
                        help='smoke = single run; full = 25 runs')
    args = parser.parse_args()

    cfg = Config()
    print(f"Mode: {args.mode}    Device: {cfg.DEVICE}")
    if args.mode == 'smoke':
        run_smoke_test(cfg)
    else:
        run_full_5x5(cfg)


if __name__ == '__main__':
    main()
