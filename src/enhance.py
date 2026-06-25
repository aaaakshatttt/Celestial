# src/enhance.py
# PURPOSE: Take the preprocessed IR images from Phase 2 and enhance them.
# This improves contrast, sharpens edges, and boosts structural detail
# so the colorization model has cleaner input to work with.

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm   # tqdm shows a progress bar in the terminal

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
TRAIN_IR = "data/pairs/train/ir"
VAL_IR   = "data/pairs/val/ir"

# Enhancement strength controls — tweak these to get better results
CLAHE_CLIP_LIMIT    = 2.0   # Higher = more contrast boost (try 1.0 to 4.0)
CLAHE_TILE_SIZE     = 8     # Grid size for CLAHE (8 means 8x8 grid of tiles)
SHARPEN_STRENGTH    = 1.5   # How strongly to sharpen (1.0 = subtle, 2.0 = strong)
EDGE_BOOST_STRENGTH = 0.3   # How much edge detail to add (0.1 to 0.5)


# ─────────────────────────────────────────────
# TECHNIQUE 1: CLAHE
# ─────────────────────────────────────────────
def apply_clahe(image):
    """
    CLAHE = Contrast Limited Adaptive Histogram Equalization

    WHAT IT DOES:
    A normal IR satellite image is often flat and washed out — dark areas
    are too dark and bright areas too bright. CLAHE fixes this by dividing
    the image into small tiles and improving contrast in each tile separately.
    The "Contrast Limited" part prevents it from over-amplifying noise.

    WHY WE NEED IT:
    Better contrast means the model can see roads, buildings, water, and
    vegetation much more clearly.

    INPUT:  float32 image in range [0, 1], shape (256, 256)
    OUTPUT: float32 image in range [0, 1], shape (256, 256)
    """
    # CLAHE in OpenCV works on 8-bit images (0-255), not float (0-1)
    # So we temporarily convert to uint8, apply CLAHE, then convert back
    image_uint8 = (image * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE)
    )

    enhanced = clahe.apply(image_uint8)

    # Convert back to float32 in range [0, 1]
    return enhanced.astype(np.float32) / 255.0


# ─────────────────────────────────────────────
# TECHNIQUE 2: SHARPENING (Unsharp Masking)
# ─────────────────────────────────────────────
def apply_sharpening(image):
    """
    Unsharp Masking — the most common sharpening technique.

    HOW IT WORKS:
    1. Create a blurred (soft) version of the image
    2. Subtract the blur from the original → this gives you the "detail layer"
    3. Add the detail layer back, amplified → sharper image

    Formula: sharpened = original + strength × (original - blurred)

    WHY WE NEED IT:
    IR images are naturally soft. Sharpening makes building edges, road
    boundaries, and coastlines crisper and easier for the model to learn from.

    INPUT:  float32 image in range [0, 1], shape (256, 256)
    OUTPUT: float32 image in range [0, 1], shape (256, 256)
    """
    # Step 1: Create a Gaussian blurred version (sigma=1.0 = mild blur)
    blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)

    # Step 2: Calculate the detail (high-frequency) layer
    detail = image - blurred

    # Step 3: Add amplified detail back to the original
    sharpened = image + SHARPEN_STRENGTH * detail

    # Clip to valid range — sharpening can push values outside [0, 1]
    return np.clip(sharpened, 0.0, 1.0)


# ─────────────────────────────────────────────
# TECHNIQUE 3: EDGE BOOST (Laplacian)
# ─────────────────────────────────────────────
def apply_edge_boost(image):
    """
    Laplacian Edge Boost — highlights boundaries between regions.

    HOW IT WORKS:
    The Laplacian filter detects rapid changes in pixel intensity.
    These rapid changes are exactly where edges are (e.g. where
    a road meets vegetation, or water meets land).
    We add a small amount of this edge map back to the image.

    WHY WE NEED IT:
    It helps the model better separate different land cover types —
    water vs vegetation, roads vs buildings. This helps preserve
    semantic integrity (Step 5 of the project).

    INPUT:  float32 image in range [0, 1], shape (256, 256)
    OUTPUT: float32 image in range [0, 1], shape (256, 256)
    """
    # Convert to uint8 for Laplacian (OpenCV requirement)
    image_uint8 = (image * 255).astype(np.uint8)

    # Apply Laplacian filter — returns edge map
    edges = cv2.Laplacian(image_uint8, cv2.CV_64F)

    # Normalize edges to [0, 1]
    edges = np.abs(edges)
    if edges.max() > 0:
        edges = edges / edges.max()

    edges = edges.astype(np.float32)

    # Add edge detail to original image
    boosted = image + EDGE_BOOST_STRENGTH * edges

    return np.clip(boosted, 0.0, 1.0)


# ─────────────────────────────────────────────
# COMBINE ALL THREE TECHNIQUES
# ─────────────────────────────────────────────
def enhance_ir(image):
    """
    Applies all three enhancement techniques in sequence.

    Order matters:
    1. CLAHE first  — fix contrast before sharpening
    2. Sharpen      — now sharpen the contrast-corrected image
    3. Edge boost   — finally add edge detail on top

    INPUT:  float32 array, shape (256, 256, 1)  ← has channel dim
    OUTPUT: float32 array, shape (256, 256, 1)  ← same shape back
    """
    # Remove the channel dimension for processing: (256,256,1) → (256,256)
    img = image[:, :, 0]

    # Apply the three techniques in order
    img = apply_clahe(img)
    img = apply_sharpening(img)
    img = apply_edge_boost(img)

    # Add the channel dimension back: (256,256) → (256,256,1)
    return np.expand_dims(img, axis=-1)


# ─────────────────────────────────────────────
# PROCESS ALL IR FILES IN A FOLDER
# ─────────────────────────────────────────────
def enhance_folder(folder_path):
    """
    Loads every .npy IR file in a folder, enhances it,
    and saves it back (overwriting the original).

    WHY overwrite?
    The enhanced version is strictly better for training.
    We don't need to keep the unenhanced version.
    """
    files = sorted(glob(os.path.join(folder_path, "*.npy")))

    if len(files) == 0:
        print(f"  No .npy files found in {folder_path}")
        return

    print(f"  Enhancing {len(files)} files in {folder_path}...")

    # tqdm wraps the list and shows a live progress bar
    for filepath in tqdm(files, desc="  Enhancing"):
        ir = np.load(filepath)          # Load: shape (256, 256, 1)
        ir_enhanced = enhance_ir(ir)    # Enhance
        np.save(filepath, ir_enhanced)  # Save back to same path


# ─────────────────────────────────────────────
# VISUALIZATION — compare before and after
# ─────────────────────────────────────────────
def visualize_enhancement(ir_path):
    """
    Shows the IR image BEFORE and AFTER each enhancement step side by side.
    Very useful to verify your enhancements are actually helping.

    Usage:
        visualize_enhancement("data/pairs/train/ir/ir_0000.npy")
    """
    # We need to reload the ORIGINAL before overwriting
    # So call this BEFORE running enhance_folder, or keep a copy
    ir_original = np.load(ir_path)[:, :, 0]

    # Apply techniques step by step for comparison
    after_clahe  = apply_clahe(ir_original)
    after_sharp  = apply_sharpening(after_clahe)
    after_edge   = apply_edge_boost(after_sharp)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    images = [ir_original, after_clahe, after_sharp, after_edge]
    titles = ["Original IR", "After CLAHE", "After Sharpening", "After Edge Boost"]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.suptitle("IR Enhancement — Step by Step", fontsize=14)
    plt.tight_layout()
    plt.savefig("outputs/enhancement_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved to outputs/enhancement_comparison.png")


# ─────────────────────────────────────────────
# MAIN — run everything
# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    # Step 1: Show before/after comparison on first image BEFORE enhancing
    sample_path = "data/pairs/train/ir/ir_0000.npy"
    if os.path.exists(sample_path):
        print("Saving before/after comparison...")

        # Save original temporarily for side-by-side view
        original = np.load(sample_path).copy()

        visualize_enhancement(sample_path)

    # Step 2: Enhance all train IR images
    print("\nEnhancing TRAIN IR images...")
    enhance_folder(TRAIN_IR)

    # Step 3: Enhance all val IR images
    print("\nEnhancing VAL IR images...")
    enhance_folder(VAL_IR)

    print("\nPhase 3 complete! All IR images enhanced.")
    print("Check outputs/enhancement_comparison.png to see the difference.")