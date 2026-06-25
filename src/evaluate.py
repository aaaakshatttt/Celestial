# src/evaluate.py
# PURPOSE: Measures how good your colorized images are.
#
# METRICS WE IMPLEMENT:
# 1. PSNR  — Peak Signal-to-Noise Ratio     (higher = better, in dB)
# 2. SSIM  — Structural Similarity Index    (higher = better, max=1.0)
# 3. FID   — Fréchet Inception Distance     (lower = better)
# 4. Inference Time — how fast the model runs per image

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from torch.utils.data import DataLoader

from model import Generator
from dataset import IRRGBDataset


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CHECKPOINT_PATH = "outputs/checkpoints"   # folder with saved .pth files
RESULTS_DIR     = "outputs/final"
IMAGE_SIZE      = 256


# ─────────────────────────────────────────────
# HELPER: Load the latest checkpoint
# ─────────────────────────────────────────────
def load_latest_checkpoint(G, device):
    """
    Finds and loads the most recent checkpoint file.
    Returns False if no checkpoint exists yet.
    """
    from glob import glob

    files = sorted(glob(os.path.join(CHECKPOINT_PATH, "*.pth")))
    if len(files) == 0:
        print("No checkpoint found. Train the model first!")
        return False

    latest = files[-1]
    checkpoint = torch.load(latest, map_location=device)
    G.load_state_dict(checkpoint["G_state"])
    print(f"Loaded checkpoint: {latest}")
    return True


# ─────────────────────────────────────────────
# METRIC 1: PSNR
# ─────────────────────────────────────────────
def compute_psnr(real, fake):
    """
    PSNR = Peak Signal-to-Noise Ratio

    WHAT IT MEASURES:
    The ratio between the maximum possible pixel value and the
    amount of noise/error between two images.

    FORMULA: PSNR = 10 × log10(MAX² / MSE)
    where MSE = Mean Squared Error between pixels

    INTERPRETATION:
    > 40 dB  → Excellent (nearly identical images)
    30-40 dB → Good quality
    20-30 dB → Acceptable
    < 20 dB  → Poor quality

    INPUTS: numpy arrays in range [0, 1], shape (H, W, 3)
    """
    return psnr_fn(real, fake, data_range=1.0)


# ─────────────────────────────────────────────
# METRIC 2: SSIM
# ─────────────────────────────────────────────
def compute_ssim(real, fake):
    """
    SSIM = Structural Similarity Index

    WHAT IT MEASURES:
    Unlike PSNR which compares pixels independently,
    SSIM looks at PATTERNS — luminance, contrast, and structure.
    This matches better with how humans perceive image quality.

    INTERPRETATION:
    1.0   → Perfect match
    > 0.9 → Excellent
    > 0.7 → Good
    < 0.5 → Poor

    INPUTS: numpy arrays in range [0, 1], shape (H, W, 3)
    """
    return ssim_fn(real, fake,
                   data_range=1.0,
                   channel_axis=2,      # tells ssim that axis 2 is channels
                   win_size=7)          # 7×7 window for local comparison


# ─────────────────────────────────────────────
# METRIC 3: FID (Simplified)
# ─────────────────────────────────────────────
def compute_fid_simple(real_features, fake_features):
    """
    FID = Fréchet Inception Distance (Simplified version)

    WHAT IT MEASURES:
    How similar the DISTRIBUTION of generated images is to
    the distribution of real images.

    Full FID uses a pretrained Inception network to extract features.
    Our simplified version uses pixel statistics instead —
    accurate enough for a hackathon.

    INTERPRETATION:
    < 10   → Excellent (hard to tell apart from real)
    10-50  → Good
    50-100 → Acceptable
    > 100  → Poor

    INPUTS: flattened feature vectors from real and fake images
    """
    # Calculate mean and covariance of real and fake distributions
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # FID formula: ||mu_r - mu_f||² + Tr(Σr + Σf - 2√(ΣrΣf))
    diff = mu_real - mu_fake
    mean_diff = np.dot(diff, diff)

    # Matrix square root approximation
    covmean = np.sqrt(np.abs(sigma_real * sigma_fake))

    fid = mean_diff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return float(np.real(fid))


# ─────────────────────────────────────────────
# METRIC 4: INFERENCE TIME
# ─────────────────────────────────────────────
def compute_inference_time(G, device, n_runs=50):
    """
    Measures how long the model takes to colorize one image.

    WHY IT MATTERS:
    For real-world satellite applications, speed matters.
    A model that takes 10 seconds per image is too slow
    for real-time monitoring.

    We run 50 times and take the average to get a stable measurement.
    """
    G.eval()
    dummy = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE).to(device)

    # Warmup run — first run is always slower due to cache
    with torch.no_grad():
        _ = G(dummy)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = G(dummy)
            end = time.perf_counter()
            times.append(end - start)

    avg_ms = np.mean(times) * 1000   # Convert to milliseconds
    return avg_ms


# ─────────────────────────────────────────────
# FULL EVALUATION PIPELINE
# ─────────────────────────────────────────────
def evaluate():
    """
    Runs complete evaluation on the validation set.
    Computes all 4 metrics and saves visual results.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}\n")

    # ── Load model ──
    G = Generator().to(device)
    loaded = load_latest_checkpoint(G, device)
    if not loaded:
        return
    G.eval()

    # ── Load validation data ──
    val_dataset = IRRGBDataset(
        ir_dir  = "data/pairs/val/ir",
        rgb_dir = "data/pairs/val/rgb"
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # ── Collect metrics ──
    psnr_scores  = []
    ssim_scores  = []
    real_feats   = []
    fake_feats   = []

    print("Running evaluation on validation set...")

    for ir, real_rgb in tqdm(val_loader, desc="Evaluating"):
        ir       = ir.to(device)
        real_rgb = real_rgb.to(device)

        with torch.no_grad():
            fake_rgb = G(ir)

        # Convert tensors to numpy images in range [0, 1]
        def tensor_to_numpy(t):
            img = t[0].cpu().numpy()           # (C, H, W)
            img = np.transpose(img, (1, 2, 0)) # (H, W, C)
            img = (img + 1.0) / 2.0            # [-1,1] → [0,1]
            return np.clip(img, 0, 1)

        real_np = tensor_to_numpy(real_rgb)
        fake_np = tensor_to_numpy(fake_rgb)

        # Compute PSNR and SSIM
        psnr_scores.append(compute_psnr(real_np, fake_np))
        ssim_scores.append(compute_ssim(real_np, fake_np))

        # Collect flattened features for FID
        real_feats.append(real_np.flatten())
        fake_feats.append(fake_np.flatten())

    # ── Compute FID ──
    real_feats = np.array(real_feats)
    fake_feats = np.array(fake_feats)
    fid_score  = compute_fid_simple(real_feats, fake_feats)

    # ── Compute Inference Time ──
    inf_time = compute_inference_time(G, device)

    # ── Print Results ──
    print("\n" + "="*45)
    print("         EVALUATION RESULTS")
    print("="*45)
    print(f"  PSNR  (higher is better) : {np.mean(psnr_scores):.4f} dB")
    print(f"  SSIM  (higher is better) : {np.mean(ssim_scores):.4f}")
    print(f"  FID   (lower  is better) : {fid_score:.4f}")
    print(f"  Inference Time           : {inf_time:.2f} ms/image")
    print("="*45)

    # ── Save results to text file ──
    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*40 + "\n")
        f.write(f"PSNR  : {np.mean(psnr_scores):.4f} dB\n")
        f.write(f"SSIM  : {np.mean(ssim_scores):.4f}\n")
        f.write(f"FID   : {fid_score:.4f}\n")
        f.write(f"Inference Time : {inf_time:.2f} ms/image\n")
    print(f"\nMetrics saved to {RESULTS_DIR}/metrics.txt")

    # ── Save visual comparison grid ──
    save_visual_results(G, val_loader, device)


# ─────────────────────────────────────────────
# SAVE VISUAL RESULTS
# ─────────────────────────────────────────────
def save_visual_results(G, val_loader, device, n_samples=5):
    """
    Saves a grid of n_samples showing:
    IR input | Generated RGB | Real RGB

    This is what you'll show in your hackathon presentation.
    """
    G.eval()
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    axes[0][0].set_title("IR Input",        fontsize=13)
    axes[0][1].set_title("Generated RGB",   fontsize=13)
    axes[0][2].set_title("Real RGB Target", fontsize=13)

    for i, (ir, real_rgb) in enumerate(val_loader):
        if i >= n_samples:
            break

        ir       = ir.to(device)
        real_rgb = real_rgb.to(device)

        with torch.no_grad():
            fake_rgb = G(ir)

        def to_img(t, gray=False):
            img = t[0].cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            img = (img + 1.0) / 2.0
            img = np.clip(img, 0, 1)
            return img[:, :, 0] if gray else img

        axes[i][0].imshow(to_img(ir, gray=True), cmap="gray")
        axes[i][1].imshow(to_img(fake_rgb))
        axes[i][2].imshow(to_img(real_rgb))

        for ax in axes[i]:
            ax.axis("off")

    plt.suptitle("IR → RGB Colorization Results", fontsize=15)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "visual_results.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visual results saved to {save_path}")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    evaluate()