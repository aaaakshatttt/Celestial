# src/inference.py
# PURPOSE: Run the trained model on ANY new IR image and save the colorized result.
#
# Use this after training is complete to:
# 1. Colorize a single IR image
# 2. Colorize an entire folder of IR images
# 3. Save side-by-side comparison images for presentation

import os
import time
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from glob import glob

from model import Generator


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CHECKPOINT_PATH = "outputs/checkpoints"   # folder with saved .pth files
OUTPUT_DIR      = "outputs/final"
IMAGE_SIZE      = 256


# ─────────────────────────────────────────────
# STEP 1: LOAD THE TRAINED MODEL
# ─────────────────────────────────────────────
def load_model(device):
    """
    Loads the trained Generator from the latest checkpoint.

    WHY latest?
    The last checkpoint has the most training — it should
    produce the best results.
    """
    files = sorted(glob(os.path.join(CHECKPOINT_PATH, "*.pth")))

    if len(files) == 0:
        raise FileNotFoundError(
            "No checkpoint found!\n"
            "Please train the model first by running: python src/train.py"
        )

    latest = files[-1]
    print(f"Loading model from: {latest}")

    checkpoint = torch.load(latest, map_location=device)

    G = Generator().to(device)
    G.load_state_dict(checkpoint["G_state"])
    G.eval()   # Set to evaluation mode — disables Dropout

    print("Model loaded successfully!\n")
    return G


# ─────────────────────────────────────────────
# STEP 2: PREPROCESS INPUT IMAGE
# ─────────────────────────────────────────────
def preprocess_image(image_path):
    """
    Loads and prepares a single IR image for inference.

    Accepts:
    - .npy files  (from your preprocessed dataset)
    - .png files  (standard image format)
    - .jpg files  (standard image format)
    - .tif files  (raw satellite files)

    Returns a PyTorch tensor ready for the model.
    shape: (1, 1, 256, 256) — batch of 1, single channel
    """
    ext = os.path.splitext(image_path)[1].lower()

    if ext == ".npy":
        # Load preprocessed numpy file
        img = np.load(image_path)           # (256, 256, 1)
        img = img[:, :, 0]                  # → (256, 256)

    elif ext in [".png", ".jpg", ".jpeg"]:
        # Load standard image and convert to grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE),
                         interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0   # Normalize to [0, 1]

    elif ext in [".tif", ".tiff"]:
        # Load raw satellite TIF file
        import rasterio
        with rasterio.open(image_path) as src:
            img = src.read(1).astype(np.float32)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE),
                         interpolation=cv2.INTER_AREA)
        if img.max() > 0:
            img = img / img.max()              # Normalize to [0, 1]

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Scale from [0,1] to [-1,1] — matches training normalization
    img = img * 2.0 - 1.0

    # Convert to tensor: (256,256) → (1, 1, 256, 256)
    tensor = torch.from_numpy(img).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0)   # Add batch + channel dims

    return tensor, img


# ─────────────────────────────────────────────
# STEP 3: RUN INFERENCE
# ─────────────────────────────────────────────
def colorize(G, ir_tensor, device):
    """
    Passes the IR image through the Generator and returns
    the colorized RGB image as a numpy array.

    Returns:
        rgb_np : numpy array, shape (256, 256, 3), range [0, 1]
    """
    ir_tensor = ir_tensor.to(device)

    start = time.perf_counter()

    with torch.no_grad():       # No gradient calculation needed for inference
        fake_rgb = G(ir_tensor)

    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"Inference time: {elapsed_ms:.2f} ms")

    # Convert output tensor to numpy image
    # Shape: (1, 3, 256, 256) → (256, 256, 3)
    rgb_np = fake_rgb[0].cpu().numpy()
    rgb_np = np.transpose(rgb_np, (1, 2, 0))   # (C,H,W) → (H,W,C)
    rgb_np = (rgb_np + 1.0) / 2.0              # [-1,1] → [0,1]
    rgb_np = np.clip(rgb_np, 0, 1)

    return rgb_np


# ─────────────────────────────────────────────
# STEP 4: SAVE RESULTS
# ─────────────────────────────────────────────
def save_result(ir_raw, rgb_np, output_path, filename):
    """
    Saves three things:
    1. Side-by-side comparison image (for presentation)
    2. The colorized RGB image alone (for further use)
    """
    os.makedirs(output_path, exist_ok=True)

    # ── Side-by-side comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # IR input (convert from [-1,1] back to [0,1] for display)
    ir_display = (ir_raw + 1.0) / 2.0
    axes[0].imshow(ir_display, cmap="gray")
    axes[0].set_title("Infrared Input", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(rgb_np)
    axes[1].set_title("Colorized RGB Output", fontsize=14)
    axes[1].axis("off")

    plt.suptitle("IR → RGB Colorization", fontsize=16)
    plt.tight_layout()

    comparison_path = os.path.join(output_path, f"{filename}_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison saved: {comparison_path}")

    # ── RGB image alone ──
    rgb_uint8 = (rgb_np * 255).astype(np.uint8)
    rgb_bgr   = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)  # OpenCV uses BGR

    rgb_path = os.path.join(output_path, f"{filename}_colorized.png")
    cv2.imwrite(rgb_path, rgb_bgr)
    print(f"Colorized image saved: {rgb_path}")


# ─────────────────────────────────────────────
# COLORIZE A SINGLE IMAGE
# ─────────────────────────────────────────────
def run_single(image_path):
    """
    Colorizes one IR image and saves the result.

    Usage:
        python src/inference.py
        (edit image_path below to point to your image)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    G = load_model(device)

    # Preprocess
    print(f"Processing: {image_path}")
    ir_tensor, ir_raw = preprocess_image(image_path)

    # Colorize
    rgb_np = colorize(G, ir_tensor, device)

    # Save
    filename = os.path.splitext(os.path.basename(image_path))[0]
    save_result(ir_raw, rgb_np, OUTPUT_DIR, filename)

    print(f"\nDone! Check outputs/final/ for results.")


# ─────────────────────────────────────────────
# COLORIZE AN ENTIRE FOLDER
# ─────────────────────────────────────────────
def run_folder(folder_path):
    """
    Colorizes ALL IR images in a folder.
    Supports .npy, .png, .jpg, .tif files.

    Usage:
        Change mode = "folder" below
        Set folder_path to your folder
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    G = load_model(device)

    # Find all supported files
    extensions = ["*.npy", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(folder_path, ext)))
    files = sorted(files)

    if len(files) == 0:
        print(f"No image files found in: {folder_path}")
        return

    print(f"Found {len(files)} images to colorize...\n")

    for i, path in enumerate(files):
        print(f"[{i+1}/{len(files)}] {os.path.basename(path)}")
        try:
            ir_tensor, ir_raw = preprocess_image(path)
            rgb_np = colorize(G, ir_tensor, device)
            filename = os.path.splitext(os.path.basename(path))[0]
            save_result(ir_raw, rgb_np, OUTPUT_DIR, filename)
        except Exception as e:
            print(f"  Error: {e} — skipping")

    print(f"\nAll done! {len(files)} images colorized.")
    print(f"Results saved to: {OUTPUT_DIR}")


# ─────────────────────────────────────────────
# DEMO — colorize validation samples
# ─────────────────────────────────────────────
def run_demo():
    """
    Colorizes the first 5 validation IR images.
    Great for testing without needing new data.
    Use this to generate images for your presentation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = load_model(device)

    val_files = sorted(glob("data/pairs/val/ir/*.npy"))[:5]

    if len(val_files) == 0:
        print("No validation files found. Run preprocess.py first.")
        return

    print(f"Running demo on {len(val_files)} validation images...\n")

    all_results = []

    for path in val_files:
        ir_tensor, ir_raw = preprocess_image(path)
        rgb_np = colorize(G, ir_tensor, device)
        all_results.append((ir_raw, rgb_np))

    # Save a big comparison grid — perfect for presentation
    fig, axes = plt.subplots(len(val_files), 2,
                             figsize=(10, 5 * len(val_files)))

    for i, (ir_raw, rgb_np) in enumerate(all_results):
        ir_display = (ir_raw + 1.0) / 2.0
        axes[i][0].imshow(ir_display, cmap="gray")
        axes[i][0].set_title(f"IR Input {i+1}",        fontsize=12)
        axes[i][0].axis("off")

        axes[i][1].imshow(rgb_np)
        axes[i][1].set_title(f"Colorized Output {i+1}", fontsize=12)
        axes[i][1].axis("off")

    plt.suptitle("IR → RGB Colorization Demo", fontsize=16)
    plt.tight_layout()

    demo_path = os.path.join(OUTPUT_DIR, "demo_grid.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(demo_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nDemo grid saved to: {demo_path}")
    print("Use this image in your hackathon presentation!")


# ─────────────────────────────────────────────
# RUN — choose your mode here
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # ── CHOOSE ONE MODE ──────────────────────
    MODE = "demo"    # Options: "single" | "folder" | "demo"
    # ─────────────────────────────────────────

    if MODE == "single":
        # Change this path to your IR image
        IMAGE_PATH = "data/pairs/val/ir/ir_0000.npy"
        run_single(IMAGE_PATH)

    elif MODE == "folder":
        # Change this to your folder of IR images
        FOLDER_PATH = "data/pairs/val/ir"
        run_folder(FOLDER_PATH)

    elif MODE == "demo":
        # Colorizes first 5 val images + saves a presentation grid
        run_demo()
        