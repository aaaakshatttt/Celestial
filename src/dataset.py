# src/dataset.py
# PURPOSE: Tells PyTorch HOW to load your IR and RGB .npy files
#          during model training. PyTorch calls this automatically
#          in batches while training.

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob


# ─────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────
class IRRGBDataset(Dataset):
    """
    Custom PyTorch Dataset for IR → RGB pairs.

    WHAT IS A DATASET CLASS?
    PyTorch needs a standard way to access your data.
    You give it this class and it handles:
    - Loading files one by one
    - Shuffling during training
    - Loading in parallel (multiple workers)
    - Grouping into batches (e.g. 4 images at a time)

    You MUST implement 3 methods:
    1. __init__  → setup / find files
    2. __len__   → how many samples total?
    3. __getitem__ → load and return ONE sample by index
    """

    def __init__(self, ir_dir, rgb_dir):
        """
        ir_dir  : folder containing ir_0000.npy, ir_0001.npy ...
        rgb_dir : folder containing rgb_0000.npy, rgb_0001.npy ...
        """
        self.ir_files  = sorted(glob(os.path.join(ir_dir,  "*.npy")))
        self.rgb_files = sorted(glob(os.path.join(rgb_dir, "*.npy")))

        # Safety check — both folders must have same number of files
        assert len(self.ir_files) == len(self.rgb_files), \
            f"Mismatch! {len(self.ir_files)} IR files but {len(self.rgb_files)} RGB files."

        print(f"Dataset loaded: {len(self.ir_files)} pairs from {ir_dir}")

    def __len__(self):
        """Returns total number of image pairs."""
        return len(self.ir_files)

    def __getitem__(self, index):
        """
        Loads ONE IR+RGB pair by index and returns as PyTorch tensors.

        WHAT IS A TENSOR?
        A tensor is just a NumPy array but for PyTorch.
        PyTorch models only accept tensors, not NumPy arrays.

        SHAPE CONVENTION in PyTorch:
        PyTorch expects images as (Channels, Height, Width)
        But our .npy files are  (Height, Width, Channels)
        So we use np.transpose to swap the order.

        IR  input : (256, 256, 1) → (1, 256, 256)
        RGB target: (256, 256, 3) → (3, 256, 256)
        """
        # Load the numpy arrays
        ir  = np.load(self.ir_files[index])   # (256, 256, 1)
        rgb = np.load(self.rgb_files[index])  # (256, 256, 3)

        # Convert (H, W, C) → (C, H, W) for PyTorch
        ir  = np.transpose(ir,  (2, 0, 1))   # → (1, 256, 256)
        rgb = np.transpose(rgb, (2, 0, 1))   # → (3, 256, 256)

        # Convert to PyTorch tensors
        ir  = torch.from_numpy(ir).float()
        rgb = torch.from_numpy(rgb).float()

        # Scale from [0,1] to [-1,1]
        # WHY: Pix2Pix uses Tanh activation at output (range -1 to 1)
        # so targets must also be in that range
        ir  = ir  * 2.0 - 1.0
        rgb = rgb * 2.0 - 1.0

        return ir, rgb


# ─────────────────────────────────────────────
# DATALOADER HELPER
# ─────────────────────────────────────────────
def get_dataloaders(batch_size=4):
    """
    Creates DataLoaders for train and val sets.

    WHAT IS A DATALOADER?
    A DataLoader wraps your Dataset and:
    - Loads images in groups (batches) of batch_size
    - Shuffles the training data every epoch
    - Can load data in parallel using multiple CPU workers

    batch_size=4 means the model sees 4 image pairs at once.
    Larger batch = more GPU memory needed.
    Start with 4, reduce to 2 if you get memory errors.
    """
    train_dataset = IRRGBDataset(
        ir_dir  = "data/pairs/train/ir",
        rgb_dir = "data/pairs/train/rgb"
    )
    val_dataset = IRRGBDataset(
        ir_dir  = "data/pairs/val/ir",
        rgb_dir = "data/pairs/val/rgb"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,        # Shuffle training data every epoch
        num_workers=0,       # 0 = load on main thread (safe for Windows)
        pin_memory=True      # Faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,       # Don't shuffle validation
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader


# ─────────────────────────────────────────────
# TEST THE DATASET
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing dataset...")

    train_loader, val_loader = get_dataloaders(batch_size=4)

    # Load one batch and print shapes
    ir_batch, rgb_batch = next(iter(train_loader))

    print(f"\nOne batch loaded successfully!")
    print(f"IR  batch shape : {ir_batch.shape}")   # (4, 1, 256, 256)
    print(f"RGB batch shape : {rgb_batch.shape}")  # (4, 3, 256, 256)
    print(f"IR  value range : {ir_batch.min():.2f} to {ir_batch.max():.2f}")
    print(f"RGB value range : {rgb_batch.min():.2f} to {rgb_batch.max():.2f}")
    print(f"\nTrain batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")
    print("\nDataset is ready for training!")