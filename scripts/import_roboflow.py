#!/usr/bin/env python3
"""
Copies a Roboflow YOLO export into dataset/images and dataset/labels.

Usage (from the repo root):
    python scripts/import_roboflow.py path/to/Swimmer-eval.v1.yolov11.zip

Roboflow exports split images into train/valid/test folders. This is an
evaluation set (we never train on it), so the script flattens all splits
into one folder. Every copied file gets an "rf_" prefix so it can't clash
with the existing swimmer_eval_XXX files.

It also checks that every label line uses class 0 (swimmer), and stops if
it finds anything else.
"""

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE_DIR / "dataset" / "images"
LABELS_DIR = BASE_DIR / "dataset" / "labels"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    src = Path(sys.argv[1]).expanduser().resolve()
    if not src.exists():
        sys.exit(f"Not found: {src}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        if src.is_file():
            with zipfile.ZipFile(src) as z:
                z.extractall(tmp)
            root = Path(tmp)
        else:
            root = src  # already-unzipped folder

        # First pass: read everything and check the classes, without copying
        pairs = []
        bad_classes = set()
        for img in sorted(root.rglob("*")):
            if img.suffix.lower() not in IMAGE_EXTS or img.parent.name != "images":
                continue
            label = img.parent.parent / "labels" / (img.stem + ".txt")
            lines = []
            if label.exists():
                lines = [l for l in label.read_text().splitlines() if l.strip()]
            for l in lines:
                if l.split()[0] != "0":
                    bad_classes.add(l.split()[0])
            pairs.append((img, lines))

        if bad_classes:
            sys.exit(
                f"Found class ids other than 0: {sorted(bad_classes)}. "
                "Merge all classes into 'swimmer' in Roboflow and export again. "
                "Nothing was copied."
            )
        if not pairs:
            sys.exit("No images found in the export. Nothing was copied.")

        # Second pass: copy
        boxes = 0
        empty = 0
        for img, lines in pairs:
            new_stem = "rf_" + img.stem
            shutil.copy2(img, IMAGES_DIR / (new_stem + img.suffix.lower()))
            (LABELS_DIR / (new_stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
            boxes += len(lines)
            empty += 0 if lines else 1
        copied = len(pairs)

    print(f"Copied {copied} images ({boxes} swimmer boxes, {empty} with no swimmers)")
    print(f"  images -> {IMAGES_DIR}")
    print(f"  labels -> {LABELS_DIR}")


if __name__ == "__main__":
    main()
