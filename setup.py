# run_setup.py  — run this once to create all folders
import os

folders = [
    "data/raw", "data/processed",
    "data/pairs/train/ir", "data/pairs/train/rgb",
    "data/pairs/val/ir", "data/pairs/val/rgb",
    "src", "notebooks",
    "outputs/checkpoints", "outputs/samples", "outputs/final",
    "presentation"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created: {folder}")

print("\nProject structure ready!")