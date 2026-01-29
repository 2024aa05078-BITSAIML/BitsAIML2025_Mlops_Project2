import os
import random
from pathlib import Path
from PIL import Image

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

IMAGE_SIZE = (128, 128)
TRAIN_SPLIT = 0.8
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

random.seed(42)


def collect_all_files(folder):
    files = []
    for root, _, filenames in os.walk(folder):
        for f in filenames:
            files.append(Path(root) / f)
    return files


def preprocess_and_split(raw_dir, processed_dir):
    print("\nStarting data preprocessing...\n")

    classes = [d for d in raw_dir.iterdir() if d.is_dir()]
    print("Detected classes:", [c.name for c in classes], "\n")

    total_images = 0

    for split in ["train", "val"]:
        for cls in classes:
            (processed_dir / split / cls.name).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        print(f"Scanning class folder: {cls}")

        all_files = collect_all_files(cls)
        print(f"  Total files found (any type): {len(all_files)}")

        image_files = [
            f for f in all_files if f.suffix in VALID_EXTENSIONS
        ]

        print(f"  Valid image files found: {len(image_files)}")

        if image_files:
            print("  Sample files:")
            for f in image_files[:3]:
                print("   ", f)

        if not image_files:
            continue

        random.shuffle(image_files)
        split_idx = int(len(image_files) * TRAIN_SPLIT)

        train_imgs = image_files[:split_idx]
        val_imgs = image_files[split_idx:]

        for img in train_imgs:
            save_image(img, processed_dir / "train" / cls.name)

        for img in val_imgs:
            save_image(img, processed_dir / "val" / cls.name)

        total_images += len(image_files)

    print("\n-----------------------------------")
    print("Total images processed:", total_images)
    print("-----------------------------------")

    if total_images == 0:
        raise ValueError(
            "\nNo images detected.\n"
            "👉 VERY LIKELY causes:\n"
            "1. Images are inside extra subfolders (train/cats/...)\n"
            "2. Dataset zip not extracted\n"
            "3. Files are not images\n\n"
            "Run:\n"
            "dir data\\raw\\cats -Recurse\n"
            "dir data\\raw\\dogs -Recurse\n"
        )

    print("\n✅ Data preprocessing completed successfully!")


def save_image(src, dest_dir):
    try:
        with Image.open(src) as img:
            img = img.convert("RGB")
            img = img.resize(IMAGE_SIZE)
            img.save(dest_dir / src.name)
    except Exception as e:
        print(f"⚠️ Skipping corrupted file: {src} | {e}")


if __name__ == "__main__":
    preprocess_and_split(RAW_DATA_DIR, PROCESSED_DATA_DIR)
