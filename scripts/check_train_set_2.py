#!/usr/bin/env python3
"""
Checks swimmer_train_2.zip (a Roboflow YOLOv11 export) against the
checklist in "How to build a second swimmer training set" before it's sent.

It checks:
  * one class only, named "person"
  * the train/valid split is about 80/20 and there's no test split
  * how many images, boxes and empty (null) images there are (~150 empty)
  * no images squashed to a square by a Resize step
  * no augmented copies (Roboflow adds "_aug" or several copies of one image)
  * no image is a near-copy of an eval image in dataset/images

Usage:
    python scripts/check_train_set_2.py ~/Downloads/swimmer_train_2.zip
"""

import re
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_IMAGES = BASE_DIR / "dataset" / "images"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dhash(img, size=8):
    """Same near-duplicate hash as build_train_set.py."""
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    px = list(g.getdata())
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (px[r * (size + 1) + c] > px[r * (size + 1) + c + 1])
    return bits


def original_name(stem):
    """Roboflow names files <original>_<ext>.rf.<hash>; strip the hash part."""
    return re.sub(r"\.rf\.[0-9a-f]+$", "", stem)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    zip_path = Path(sys.argv[1]).expanduser()
    problems, notes = [], []

    if zip_path.name != "swimmer_train_2.zip":
        notes.append(f"File is called {zip_path.name}; the guide says swimmer_train_2.zip")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(root)

        # ---- classes ----
        yamls = list(root.rglob("data.yaml"))
        if not yamls:
            problems.append("No data.yaml: was it exported in YOLOv11 format?")
            names = []
        else:
            names = yaml.safe_load(yamls[0].read_text()).get("names", [])
            names = list(names.values()) if isinstance(names, dict) else list(names)
            if names != ["person"]:
                problems.append(f"Classes are {names}; should be exactly ['person']")

        # ---- images, boxes, splits ----
        split_count = Counter()
        boxes = 0
        empty = 0
        square = 0
        bad_class = 0
        originals = Counter()
        hashes = []
        for img in sorted(root.rglob("*")):
            if img.suffix.lower() not in IMAGE_EXTS or img.parent.name != "images":
                continue
            split = img.parent.parent.name
            split_count[split] += 1
            originals[original_name(img.stem)] += 1
            lbl = img.parent.parent / "labels" / (img.stem + ".txt")
            rows = [l.split() for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
            if not rows:
                empty += 1
            boxes += len(rows)
            bad_class += sum(1 for r in rows if r[0] != "0")
            with Image.open(img) as im:
                if im.size[0] == im.size[1] == 640:
                    square += 1
                hashes.append((img.name, dhash(im)))

        total = sum(split_count.values())
        if total == 0:
            sys.exit("No images found in the zip.")

        if bad_class:
            problems.append(f"{bad_class} boxes use a class id other than 0")
        if split_count.get("test"):
            problems.append(f"{split_count['test']} images are in a test split; the guide says 0% test")
        val_share = split_count.get("valid", 0) / total
        if not 0.12 <= val_share <= 0.28:
            problems.append(f"Valid split is {val_share:.0%}; aim for about 20%")
        if not 100 <= empty <= 250:
            notes.append(f"{empty} empty-water images; the guide asks for about 150")
        if square > 0.5 * total:
            problems.append(f"{square}/{total} images are exactly 640x640: the Resize step is probably still on")
        dupes = {k: v for k, v in originals.items() if v > 1}
        if dupes:
            problems.append(f"{len(dupes)} images appear more than once (augmented copies?), e.g. {next(iter(dupes))}")
        if total < 800:
            notes.append(f"Only {total} images; the goal is 800-1,500")

        # ---- overlap with the eval set ----
        eval_hashes = []
        for p in sorted(EVAL_IMAGES.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                with Image.open(p) as im:
                    eval_hashes.append((p.name, dhash(im)))
        overlap = []
        for name, h in hashes:
            for ename, eh in eval_hashes:
                if bin(h ^ eh).count("1") <= 6:
                    overlap.append((name, ename))
                    break
        if overlap:
            problems.append(f"{len(overlap)} images look like eval images (delete them in Roboflow):")
            problems += [f"    {a}  ~  {b}" for a, b in overlap[:30]]

    print(f"\n{zip_path.name}")
    print(f"  images: {total}  (train {split_count.get('train', 0)}, valid {split_count.get('valid', 0)}, "
          f"test {split_count.get('test', 0)})")
    print(f"  person boxes: {boxes}")
    print(f"  empty-water images: {empty}")
    print(f"  classes: {names}")
    for n in notes:
        print(f"  note: {n}")
    if problems:
        print("\nFIX BEFORE SENDING:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nAll checks passed. Still check by hand: every person in the water is boxed, nobody out of it.")


if __name__ == "__main__":
    main()
