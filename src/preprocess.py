


import os
import numpy as np
import cv2
import rasterio
import matplotlib.pyplot as plt
from glob import glob
import random
import shutil


# CONFIGURATION 
IMAGE_SIZE = 256          
TRAIN_SPLIT = 0.8         
RANDOM_SEED = 42         


RAW_DIR    = "data/raw"
OUTPUT_DIR = "data/pairs"
TRAIN_IR   = os.path.join(OUTPUT_DIR, "train", "ir")
TRAIN_RGB  = os.path.join(OUTPUT_DIR, "train", "rgb")
VAL_IR     = os.path.join(OUTPUT_DIR, "val", "ir")
VAL_RGB    = os.path.join(OUTPUT_DIR, "val", "rgb")



def read_band(filepath):
    """
    Reads a single Landsat band .TIF file and returns it as a NumPy array.

    Landsat files store pixel values as 16-bit integers (0 to 65535).
    We return it as float32 for easy math later.
    """
    with rasterio.open(filepath) as src:
        band = src.read(1)              # Read band 1 (only one band per file)
        band = band.astype(np.float32) # Convert to float so we can do math
    return band



def resize_image(image, size=IMAGE_SIZE):
    """
    Resizes an image (any number of channels) to size×size pixels.

    WHY: The model expects a fixed input size. 256×256 is the standard
         starting point — small enough to train fast, large enough to
         capture detail.

    cv2.INTER_AREA is best when shrinking images (avoids aliasing/blur).
    """
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)



def normalize(image, max_val=None):
    """
    Scales pixel values to the range [0.0, 1.0].

    WHY: Neural networks train much better on small numbers close to 0.
         Raw Landsat pixels go up to 65535 — way too large for a network.

    If max_val is not provided, we use the image's own maximum value.
    This is called "min-max normalization".
    """
    if max_val is None:
        max_val = image.max()

    if max_val == 0:                   # Avoid division by zero on blank images
        return image

    return image / max_val



def build_rgb(red_band, green_band, blue_band):
    """
    Stacks three separate band arrays into a single (H, W, 3) RGB image.

    Landsat stores R, G, B as separate files (B4, B3, B2).
    We combine them into one 3-channel image here.

    np.stack puts them along the last axis → shape (256, 256, 3)
    """
    # Normalize each band individually
    r = normalize(red_band)
    g = normalize(green_band)
    b = normalize(blue_band)

    # Stack into (H, W, 3)
    rgb = np.stack([r, g, b], axis=-1)
    return rgb.astype(np.float32)



def process_scene(b2_path, b3_path, b4_path, ir_path):
    """
    Given paths to 4 band files for one Landsat scene,
    returns a processed (ir_image, rgb_image) pair.

    b2 = Blue band
    b3 = Green band
    b4 = Red band
    ir_path = Thermal infrared band (B10 or B11)

    Returns:
        ir  → shape (256, 256, 1)  — single channel IR
        rgb → shape (256, 256, 3)  — 3-channel color image
    """
    # Read all 4 bands
    b2 = read_band(b2_path)    # Blue
    b3 = read_band(b3_path)    # Green
    b4 = read_band(b4_path)    # Red
    ir = read_band(ir_path)    # Infrared (thermal)

    # Resize all to 256×256
    b2 = resize_image(b2)
    b3 = resize_image(b3)
    b4 = resize_image(b4)
    ir = resize_image(ir)

    
    rgb = build_rgb(b4, b3, b2)     # Red=B4, Green=B3, Blue=B2

    
    ir = normalize(ir)
    ir = np.expand_dims(ir, axis=-1)  # Shape becomes (256, 256, 1)

    return ir, rgb



def save_pair(ir, rgb, ir_dir, rgb_dir, index):
    """
    Saves one IR image and its corresponding RGB image as .npy files.

    WHY .npy instead of .jpg?
    - .npy preserves float32 precision (no quality loss)
    - .jpg compresses and loses detail — bad for training
    - .npy loads instantly in Python with numpy.load()

    Files are named: ir_0001.npy, rgb_0001.npy, etc.
    """
    filename = f"{index:04d}"   # Zero-padded: 0001, 0002, ...
    np.save(os.path.join(ir_dir,  f"ir_{filename}.npy"),  ir)
    np.save(os.path.join(rgb_dir, f"rgb_{filename}.npy"), rgb)



def run_preprocessing():
    """
    Main function that:
    1. Scans data/raw/ for Landsat band files
    2. Processes each scene into an IR+RGB pair
    3. Splits into train/val sets
    4. Saves everything to data/pairs/
    """

    print("Starting preprocessing pipeline...")
    print(f"Looking for files in: {RAW_DIR}")

    
    b4_files = sorted(glob(os.path.join(RAW_DIR, "*_B4.TIF")))

    if len(b4_files) == 0:
        print("\nNo Landsat files found!")
        print("Generating synthetic test data instead...")
        _generate_synthetic_data()
        return

    print(f"Found {len(b4_files)} scenes to process.")

    
    random.seed(RANDOM_SEED)
    random.shuffle(b4_files)
    split_idx   = int(len(b4_files) * TRAIN_SPLIT)
    train_files = b4_files[:split_idx]
    val_files   = b4_files[split_idx:]

    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

   
    _process_split(train_files, TRAIN_IR, TRAIN_RGB, split_name="train")
    _process_split(val_files,   VAL_IR,   VAL_RGB,   split_name="val")

    print("\nPreprocessing complete!")
    print(f"Saved to: {OUTPUT_DIR}")


def _process_split(b4_files, ir_dir, rgb_dir, split_name):
    """
    Helper: processes a list of B4 file paths and saves pairs.
    """
    os.makedirs(ir_dir,  exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)

    for idx, b4_path in enumerate(b4_files):
        
        b2_path = b4_path.replace("_B4.TIF", "_B2.TIF")
        b3_path = b4_path.replace("_B4.TIF", "_B3.TIF")
        ir_path = b4_path.replace("_B4.TIF", "_B10.TIF")  # or _B11.TIF

       
        for p in [b2_path, b3_path, ir_path]:
            if not os.path.exists(p):
                print(f"  Skipping scene {idx+1} — missing file: {p}")
                continue

        try:
            ir, rgb = process_scene(b2_path, b3_path, b4_path, ir_path)
            save_pair(ir, rgb, ir_dir, rgb_dir, idx)
            print(f"  [{split_name}] Saved pair {idx+1:04d} — IR: {ir.shape}, RGB: {rgb.shape}")
        except Exception as e:
            print(f"  Error processing scene {idx+1}: {e}")


def _generate_synthetic_data(n_train=80, n_val=20):
    """
    Creates fake IR and RGB images for testing the pipeline.

    WHY: You can test your entire training pipeline without
         downloading real satellite data first. This lets you
         catch bugs early and verify shapes/dtypes are correct.

    The synthetic images are random noise — they won't produce
    good colorization, but they confirm your code runs correctly.
    """
    print(f"Generating {n_train} train + {n_val} val synthetic pairs...")

    for split_name, ir_dir, rgb_dir, count in [
        ("train", TRAIN_IR, TRAIN_RGB, n_train),
        ("val",   VAL_IR,   VAL_RGB,   n_val),
    ]:
        os.makedirs(ir_dir,  exist_ok=True)
        os.makedirs(rgb_dir, exist_ok=True)

        for i in range(count):
            
            ir  = np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 1).astype(np.float32)
            
            rgb = np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)

            save_pair(ir, rgb, ir_dir, rgb_dir, i)

        print(f"  {split_name}: {count} synthetic pairs saved.")

    print("Synthetic data ready! Replace with real Landsat data later.")



def visualize_sample(ir_path, rgb_path):
    """
    Loads one IR + RGB pair and displays them side by side.
    Call this after preprocessing to verify your data looks correct.

    Usage:
        visualize_sample("data/pairs/train/ir/ir_0001.npy",
                         "data/pairs/train/rgb/rgb_0001.npy")
    """
    ir  = np.load(ir_path)    # Shape: (256, 256, 1)
    rgb = np.load(rgb_path)   # Shape: (256, 256, 3)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(ir[:, :, 0], cmap="gray")  # IR is grayscale
    axes[0].set_title("Infrared (IR) — Input to model")
    axes[0].axis("off")

    axes[1].imshow(np.clip(rgb, 0, 1))         # RGB color image
    axes[1].set_title("RGB — Target for model")
    axes[1].axis("off")

    plt.suptitle("Sample training pair", fontsize=14)
    plt.tight_layout()
    plt.savefig("outputs/sample_pair.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved preview to outputs/sample_pair.png")


if __name__ == "__main__":
    # Create output directories
    os.makedirs("outputs", exist_ok=True)

    # Run the main preprocessing pipeline
    run_preprocessing()

    # Show a sample pair to verify everything looks right
    sample_ir  = "data/pairs/train/ir/ir_0000.npy"
    sample_rgb = "data/pairs/train/rgb/rgb_0000.npy"

    if os.path.exists(sample_ir):
        visualize_sample(sample_ir, sample_rgb)