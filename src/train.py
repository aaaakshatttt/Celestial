# src/train.py
# PURPOSE: Trains the Pix2Pix model.
# This is where the Generator and Discriminator actually LEARN.
#
# TRAINING LOOP (runs every epoch):
# Step 1: Generator creates fake RGB from IR
# Step 2: Discriminator judges real vs fake
# Step 3: Calculate losses
# Step 4: Update both networks using backpropagation

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import Generator, Discriminator, initialize_weights
from dataset import get_dataloaders


# ─────────────────────────────────────────────
# CONFIGURATION — tweak these as needed
# ─────────────────────────────────────────────
LEARNING_RATE  = 2e-4      # How fast the model learns (don't go higher)
BATCH_SIZE     = 4         # Images per batch (reduce to 2 if memory error)
NUM_EPOCHS     = 100       # Total training rounds (start with 50 to test)
LAMBDA_L1      = 100       # Weight of L1 loss vs GAN loss (paper uses 100)
SAVE_EVERY     = 10        # Save model checkpoint every N epochs
SAMPLE_EVERY   = 5         # Save sample images every N epochs

CHECKPOINT_DIR = "outputs/checkpoints"
SAMPLES_DIR    = "outputs/samples"


# ─────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────
def get_losses():
    """
    Pix2Pix uses TWO loss functions combined:

    1. BCE LOSS (Adversarial loss)
       Discriminator loss — real images should score 1, fake should score 0.
       Generator loss     — tries to make Discriminator output 1 for fakes.
       BCEWithLogitsLoss combines sigmoid + BCE in one step (more stable).

    2. L1 LOSS (Pixel loss)
       Direct pixel-by-pixel difference between fake RGB and real RGB.
       WHY: Keeps colors and structure accurate.
            Without L1, Generator might fool Discriminator but
            produce completely wrong colors.
       Multiplied by LAMBDA_L1=100 to balance with adversarial loss.
    """
    bce = nn.BCEWithLogitsLoss()
    l1  = nn.L1Loss()
    return bce, l1


# ─────────────────────────────────────────────
# SAVE / LOAD CHECKPOINTS
# ─────────────────────────────────────────────
def save_checkpoint(G, D, opt_G, opt_D, epoch):
    """
    Saves model weights so you can resume training later.

    WHY: Training 100 epochs can take hours. If your computer
    crashes or you close the terminal, checkpoints let you
    continue from where you left off instead of starting over.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch:04d}.pth")

    torch.save({
        "epoch"       : epoch,
        "G_state"     : G.state_dict(),
        "D_state"     : D.state_dict(),
        "opt_G_state" : opt_G.state_dict(),
        "opt_D_state" : opt_D.state_dict(),
    }, path)

    print(f"  Checkpoint saved: {path}")


def load_checkpoint(G, D, opt_G, opt_D, path):
    """
    Loads a saved checkpoint to resume training.
    Call this before training if you want to continue from a checkpoint.
    """
    checkpoint = torch.load(path, map_location="cpu")
    G.load_state_dict(checkpoint["G_state"])
    D.load_state_dict(checkpoint["D_state"])
    opt_G.load_state_dict(checkpoint["opt_G_state"])
    opt_D.load_state_dict(checkpoint["opt_D_state"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resumed from epoch {checkpoint['epoch']}")
    return start_epoch


# ─────────────────────────────────────────────
# SAVE SAMPLE IMAGES
# ─────────────────────────────────────────────
def save_samples(G, val_loader, device, epoch):
    """
    Runs the Generator on a few validation images and saves
    a side-by-side comparison: IR | Fake RGB | Real RGB

    WHY: Loss numbers alone don't tell you if images look good.
    Visual inspection during training is very important.
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    G.eval()   # Switch to evaluation mode (disables Dropout, BatchNorm behaves differently)

    with torch.no_grad():   # Don't calculate gradients (saves memory)
        ir, real_rgb = next(iter(val_loader))
        ir       = ir.to(device)
        real_rgb = real_rgb.to(device)
        fake_rgb = G(ir)

    # Convert tensors back to displayable images
    # Reverse the [-1,1] scaling back to [0,1]
    def to_img(tensor):
        img = tensor[0].cpu().numpy()        # Take first image in batch
        img = np.transpose(img, (1, 2, 0))  # (C,H,W) → (H,W,C)
        img = (img + 1.0) / 2.0             # [-1,1] → [0,1]
        return np.clip(img, 0, 1)

    ir_img   = to_img(ir)[:, :, 0]          # Grayscale IR
    fake_img = to_img(fake_rgb)              # Generated RGB
    real_img = to_img(real_rgb)              # Ground truth RGB

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(ir_img, cmap="gray");  axes[0].set_title("IR Input");       axes[0].axis("off")
    axes[1].imshow(fake_img);             axes[1].set_title("Generated RGB");   axes[1].axis("off")
    axes[2].imshow(real_img);             axes[2].set_title("Real RGB Target"); axes[2].axis("off")

    plt.suptitle(f"Epoch {epoch}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(SAMPLES_DIR, f"sample_epoch_{epoch:04d}.png"),
                dpi=100, bbox_inches="tight")
    plt.close()

    G.train()   # Switch back to training mode


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train():
    print("Initializing training...\n")

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if device.type == "cpu":
        print("  Note: CPU training is slow. 100 epochs may take several hours.")
        print("  Consider reducing NUM_EPOCHS to 20 for a quick test first.\n")

    # ── Data ──
    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # ── Models ──
    G = Generator().to(device)
    D = Discriminator().to(device)
    initialize_weights(G)
    initialize_weights(D)

    # ── Optimizers ──
    # Adam optimizer — works well for GANs
    # betas=(0.5, 0.999) is the Pix2Pix paper recommendation
    opt_G = torch.optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # ── Loss functions ──
    bce_loss, l1_loss = get_losses()

    # ── Track losses for plotting ──
    history = {"G_loss": [], "D_loss": []}

    # ─────────────────────────────────────────
    # MAIN TRAINING LOOP
    # ─────────────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):
        G.train()
        D.train()

        epoch_G_loss = 0.0
        epoch_D_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{NUM_EPOCHS}")

        for ir, real_rgb in loop:
            ir       = ir.to(device)
            real_rgb = real_rgb.to(device)

            # ══════════════════════════════════
            # STEP 1: TRAIN DISCRIMINATOR
            # Goal: correctly identify real=1, fake=0
            # ══════════════════════════════════

            # Generate fake RGB
            fake_rgb = G(ir)

            # Score real pair (IR + real RGB) — should be close to 1
            real_score = D(ir, real_rgb)
            real_labels = torch.ones_like(real_score)   # Target: 1
            loss_D_real = bce_loss(real_score, real_labels)

            # Score fake pair (IR + fake RGB) — should be close to 0
            # .detach() stops gradients flowing into Generator
            # (we don't want to update G when training D)
            fake_score = D(ir, fake_rgb.detach())
            fake_labels = torch.zeros_like(fake_score)  # Target: 0
            loss_D_fake = bce_loss(fake_score, fake_labels)

            # Total Discriminator loss
            loss_D = (loss_D_real + loss_D_fake) / 2

            # Update Discriminator
            opt_D.zero_grad()   # Clear old gradients
            loss_D.backward()   # Calculate new gradients
            opt_D.step()        # Update weights

            # ══════════════════════════════════
            # STEP 2: TRAIN GENERATOR
            # Goal: fool Discriminator + match real RGB pixels
            # ══════════════════════════════════

            # Re-score fake images — Generator wants D to output 1
            fake_score_for_G = D(ir, fake_rgb)
            real_labels_for_G = torch.ones_like(fake_score_for_G)

            # Adversarial loss — fool the Discriminator
            loss_G_bce = bce_loss(fake_score_for_G, real_labels_for_G)

            # L1 loss — match pixel values of real RGB
            loss_G_l1 = l1_loss(fake_rgb, real_rgb) * LAMBDA_L1

            # Total Generator loss
            loss_G = loss_G_bce + loss_G_l1

            # Update Generator
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # ── Update progress bar ──
            epoch_G_loss += loss_G.item()
            epoch_D_loss += loss_D.item()
            loop.set_postfix(G=f"{loss_G.item():.3f}", D=f"{loss_D.item():.3f}")

        # ── End of epoch — record average losses ──
        avg_G = epoch_G_loss / len(train_loader)
        avg_D = epoch_D_loss / len(train_loader)
        history["G_loss"].append(avg_G)
        history["D_loss"].append(avg_D)

        print(f"  Avg G Loss: {avg_G:.4f} | Avg D Loss: {avg_D:.4f}")

        # ── Save samples ──
        if epoch % SAMPLE_EVERY == 0:
            save_samples(G, val_loader, device, epoch)
            print(f"  Sample saved to outputs/samples/")

        # ── Save checkpoint ──
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(G, D, opt_G, opt_D, epoch)

    # ─────────────────────────────────────────
    # AFTER TRAINING: Plot loss curves
    # ─────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(history["G_loss"], label="Generator Loss",     color="blue")
    plt.plot(history["D_loss"], label="Discriminator Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/loss_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nTraining complete!")
    print("Loss curve saved to outputs/loss_curve.png")
    print("Sample images saved to outputs/samples/")
    print("Checkpoints saved to outputs/checkpoints/")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    train()