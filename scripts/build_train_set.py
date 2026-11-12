#!/usr/bin/env python3
"""
Builds the TRAINING set for fine-tuning YOLO26n on swimmers.

It never touches dataset/images or dataset/labels (the eval set). The
training images come from two public Roboflow Universe datasets (CC BY 4.0)
that were NOT used for the eval set:

  * TinyPerson (Chris D), version 3: far-away people at sea and on beaches.
    Only people in the water are kept; people on the beach are dropped,
    the same rule as the eval set. These photos are large and the people
    are tiny, so each photo is cut into a 2x2 grid of tiles. That makes
    the people twice as big relative to the image, closer to what a boat
    camera would see.
  * Swimmer detection 2 (test), version 2: lap swimmers lying flat in pools.
    Roboflow squashed these to 640x640, so they're stretched back to 16:9
    (most are video frames). Every 3rd image is kept, because neighbouring
    video frames are near-copies of each other. Underwater shots are dropped.

Then any training image that looks like an eval image (a near-duplicate)
is removed, so the eval stays fair.

Output: train_data/ with images/{train,val}, labels/{train,val} and
swimmers.yaml (one class: 0 = person, same id as the eval labels).

Usage (needs a free Roboflow API key: Roboflow -> Settings -> API Keys):
    ROBOFLOW_API_KEY=xxxx python scripts/build_train_set.py
"""

import os
import random
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_IMAGES = BASE_DIR / "dataset" / "images"
OUT = BASE_DIR / "train_data"
DL = BASE_DIR / "train_data_download"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
random.seed(0)

SOURCES = [
    # (workspace, project, version, short name)
    ("chris-d-dbyby", "tinyperson", 3, "tiny"),
    ("test-p2znd", "swimmer-detection-2", 2, "pool"),
]


# ---------- helpers ----------

def dhash(img, size=8):
    """64-bit difference hash: near-identical images get near-identical hashes."""
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    px = list(g.getdata())
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (px[r * (size + 1) + c] > px[r * (size + 1) + c + 1])
    return bits


def read_labels(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                rows.append((int(p[0]), *map(float, p[1:])))
    return rows


def write_labels(path, boxes):
    path.write_text("".join(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for x, y, w, h in boxes))


def find_pairs(root):
    """All (image, label, split) in a Roboflow YOLO export."""
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() in IMAGE_EXTS and img.parent.name == "images":
            split = img.parent.parent.name  # train / valid / test
            yield img, img.parent.parent / "labels" / (img.stem + ".txt"), split


def water_class_ids(names):
    """Which class ids are people in the water."""
    keep = {i for i, n in names.items() if any(k in n.lower() for k in ("wet", "swim", "water", "sea"))}
    if not keep:
        sys.exit(f"Couldn't tell which classes are people in the water: {names}. Nothing was built.")
    return keep


# ---------- download ----------

def download():
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("Set ROBOFLOW_API_KEY first (Roboflow -> Settings -> API Keys).")
    from roboflow import Roboflow
    rf = Roboflow(api_key=key)
    paths = {}
    for ws, proj, ver, name in SOURCES:
        loc = DL / name
        if not loc.exists():
            rf.workspace(ws).project(proj).version(ver).download("yolov11", location=str(loc))
        paths[name] = loc
    return paths


# ---------- per-source processing ----------

def process_tiny(root, emit):
    names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else names
    keep = water_class_ids(names)
    print(f"  TinyPerson classes {names} -> keeping {[names[i] for i in keep]}")
    for img_path, lbl, split in find_pairs(root):
        boxes = [b[1:] for b in read_labels(lbl) if b[0] in keep]
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        for ty in range(2):
            for tx in range(2):
                x0, y0 = tx * 0.5, ty * 0.5
                tile_boxes = []
                for cx, cy, w, h in boxes:
                    bx1, by1, bx2, by2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                    ix1, iy1 = max(bx1, x0), max(by1, y0)
                    ix2, iy2 = min(bx2, x0 + 0.5), min(by2, y0 + 0.5)
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                    if (ix2 - ix1) * (iy2 - iy1) < 0.5 * w * h:
                        continue  # mostly in another tile
                    tile_boxes.append((((ix1 + ix2) / 2 - x0) * 2, ((iy1 + iy2) / 2 - y0) * 2,
                                       (ix2 - ix1) * 2, (iy2 - iy1) * 2))
                tile = im.crop((int(x0 * W), int(y0 * H), int((x0 + 0.5) * W), int((y0 + 0.5) * H)))
                emit(f"tiny_{img_path.stem}_t{ty}{tx}", tile, tile_boxes, split)


def process_pool(root, emit):
    pairs = [p for p in find_pairs(root) if "underwater" not in p[0].name.lower()]
    for i, (img_path, lbl, split) in enumerate(pairs):
        if i % 3:
            continue
        boxes = [b[1:] for b in read_labels(lbl)]  # single class: swimmer
        im = Image.open(img_path).convert("RGB")
        if im.size[0] == im.size[1]:
            im = im.resize((1138, 640), Image.BILINEAR)  # undo the 640x640 squash
        emit(f"pool_{img_path.stem}", im, boxes, split)


# ---------- main ----------

def main():
    paths = download()

    eval_hashes = []
    for p in sorted(EVAL_IMAGES.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            eval_hashes.append(dhash(Image.open(p)))
    print(f"Loaded {len(eval_hashes)} eval images to check for overlap")

    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ("train", "val"):
        (OUT / "images" / s).mkdir(parents=True)
        (OUT / "labels" / s).mkdir(parents=True)

    stats = {"train": 0, "val": 0, "boxes": 0, "empty_skipped": 0, "eval_overlap": 0}

    def emit(stem, im, boxes, split):
        if any(bin(dhash(im) ^ h).count("1") <= 6 for h in eval_hashes):
            stats["eval_overlap"] += 1
            return
        if not boxes and random.random() > 0.2:  # keep ~20% of empty tiles as "nobody here" examples
            stats["empty_skipped"] += 1
            return
        s = "val" if split in ("valid", "test") else "train"
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
        im.save(OUT / "images" / s / f"{stem}.jpg", quality=92)
        write_labels(OUT / "labels" / s / f"{stem}.txt", boxes)
        stats[s] += 1
        stats["boxes"] += len(boxes)

    print("TinyPerson ...")
    process_tiny(paths["tiny"], emit)
    print("Swimmer detection 2 ...")
    process_pool(paths["pool"], emit)

    (OUT / "swimmers.yaml").write_text(yaml.safe_dump({
        "path": str(OUT), "train": "images/train", "val": "images/val", "names": {0: "person"},
    }))
    print(f"\nTraining set ready in {OUT}")
    print(f"  train images: {stats['train']}   val images: {stats['val']}   swimmer boxes: {stats['boxes']}")
    print(f"  removed as near-copies of eval images: {stats['eval_overlap']}")
    print(f"  empty tiles skipped: {stats['empty_skipped']}")


if __name__ == "__main__":
    main()
