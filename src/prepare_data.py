import os
import shutil
import random
from pathlib import Path

# --- CONFIGURATION ---
# Using absolute paths based on your terminal screenshot to prevent errors
RAW_DIR = Path(r"C:\Users\LenovoW10\Desktop\Dr Mirzaei PhD application\Project\data\raw")
OUT_DIR = Path(r"C:\Users\LenovoW10\Desktop\Dr Mirzaei PhD application\Project\data\processed\sdnet_split")

VAL_SPLIT = 0.15                # 15% of D+W goes to validation
SEED = 42                       # Keep this fixed for reproducibility

random.seed(SEED)

# Define the source paths inside raw/
# NOTE: If this fails, check if your folders are named 'C' and 'U' or 'Cracked' and 'Uncracked'
SOURCES = {
    "D": {"cracked": RAW_DIR / "D" / "CD", "uncracked": RAW_DIR / "D" / "UD"},
    "W": {"cracked": RAW_DIR / "W" / "CW", "uncracked": RAW_DIR / "W" / "UW"},
    "P": {"cracked": RAW_DIR / "P" / "CP", "uncracked": RAW_DIR / "P" / "UP"},
}

# Define the destination paths
DEST = {
    "train": {"cracked": OUT_DIR / "train" / "cracked", "uncracked": OUT_DIR / "train" / "uncracked"},
    "val":   {"cracked": OUT_DIR / "val" / "cracked",   "uncracked": OUT_DIR / "val" / "uncracked"},
    "test":  {"cracked": OUT_DIR / "test" / "cracked",  "uncracked": OUT_DIR / "test" / "uncracked"},
}

def setup_dirs():
    print("Creating folders...")
    for split in DEST.values():
        for path in split.values():
            path.mkdir(parents=True, exist_ok=True)

def process_surface(surface, is_test=False):
    for label in ["cracked", "uncracked"]:
        src_path = SOURCES[surface][label]
        
        if not src_path.exists():
            print(f"ERROR: Path does not exist: {src_path}")
            print("Please check if your raw data subfolders are named 'C' and 'U', or 'Cracked' and 'Uncracked'.")
            continue

        # Get all image files
        images = [f for f in src_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        random.shuffle(images)

        if is_test:
            # All Pavement (P) images go directly to test
            dest_path = DEST["test"][label]
            for img in images:
                shutil.copy(img, dest_path / img.name)
            print(f"Moved {len(images)} {label} {surface} images to TEST")
        else:
            # Split D and W images into Train and Val
            split_idx = int(len(images) * (1 - VAL_SPLIT))
            train_imgs = images[:split_idx]
            val_imgs = images[split_idx:]

            for img in train_imgs:
                shutil.copy(img, DEST["train"][label] / img.name)
            for img in val_imgs:
                shutil.copy(img, DEST["val"][label] / img.name)
            
            print(f"Split {label} {surface}: {len(train_imgs)} to TRAIN, {len(val_imgs)} to VAL")

if __name__ == "__main__":
    print("Setting up directories...")
    setup_dirs()
    
    print("\nProcessing Bridge Decks (D)...")
    process_surface("D", is_test=False)
    
    print("\nProcessing Walls (W)...")
    process_surface("W", is_test=False)
    
    print("\nProcessing Pavements (P) for Holdout Test...")
    process_surface("P", is_test=True)
    
    print(f"\nDone! Your organized dataset is in: {OUT_DIR}")