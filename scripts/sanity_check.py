#!/usr/bin/env python3
"""
Sanity check for the swimmer evaluation: are YOLO's misses real, or are they
partly a scoring artifact (YOLO finds the person, but its box doesn't overlap
the label box enough to count)?

It runs YOLO26n once per image, then reports:
  1. Recall and precision with a strict overlap rule (IoU >= 0.5, same as
     evaluate_all.py), and looser ones (0.3 and 0.1). If recall jumps a lot
     with a looser rule, many "misses" are really box-shape disagreements.
  2. Recall by swimmer size (how big the person is in the image), so we can
     see whether YOLO mostly misses small, far-away people.
  3. The same at a larger input size (1280 instead of 640), which usually
     helps with small people, at the cost of speed.
It also saves pictures of the 20 worst images, with the labels in GREEN and
YOLO's detections in RED, so we can see with our own eyes what's going on.

Usage (from the repo root):
    python scripts/sanity_check.py                  # sizes 640 and 1280
    python scripts/sanity_check.py --imgsz 640      # faster, 640 only
Outputs go to dataset/eval_results/sanity/
"""

import argparse
import csv
import os

import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "images")
LABELS_DIR = os.path.join(BASE_DIR, "dataset", "labels")
OUT_DIR = os.path.join(BASE_DIR, "dataset", "eval_results", "sanity")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

CONFS = [0.25, 0.15]
IOUS = [0.5, 0.3, 0.1]
# Swimmer size = the longer side of the label box, as a fraction of the image
SIZE_BINS = [("tiny (<3%)", 0.0, 0.03), ("small (3-6%)", 0.03, 0.06),
             ("medium (6-12%)", 0.06, 0.12), ("large (>12%)", 0.12, 9.9)]


def iou(a, b):
    """Boxes are (cx, cy, w, h), normalised 0-1."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match(gts, preds, iou_thr):
    """Greedy matching, best overlaps first. Returns (set of matched gt idx, n matched preds)."""
    pairs = sorted(((iou(p, g), pi, gi) for pi, p in enumerate(preds) for gi, g in enumerate(gts)),
                   reverse=True)
    used_p, used_g = set(), set()
    for v, pi, gi in pairs:
        if v < iou_thr:
            break
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
    return used_g, len(used_p)


def load_labels(stem):
    path = os.path.join(LABELS_DIR, stem + ".txt")
    boxes = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 5:
                    boxes.append(tuple(float(x) for x in parts[1:5]))
    return boxes


def size_bin(box):
    s = max(box[2], box[3])
    for name, lo, hi in SIZE_BINS:
        if lo <= s < hi:
            return name
    return SIZE_BINS[-1][0]


def draw(img_path, gts, preds, out_path):
    im = cv2.imread(img_path)
    h, w = im.shape[:2]
    t = max(2, w // 400)

    def rect(b, color, text=None):
        x1, y1 = int((b[0] - b[2] / 2) * w), int((b[1] - b[3] / 2) * h)
        x2, y2 = int((b[0] + b[2] / 2) * w), int((b[1] + b[3] / 2) * h)
        cv2.rectangle(im, (x1, y1), (x2, y2), color, t)
        if text:
            cv2.putText(im, text, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.4, w / 1600), color, max(1, t // 2))

    for g in gts:
        rect(g, (0, 200, 0))                     # labels: green
    for p, c in preds:
        rect(p, (0, 0, 255), f"{c:.2f}")         # YOLO: red, with confidence
    cv2.imwrite(out_path, im)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imgsz", default="640,1280", help="comma-separated input sizes")
    ap.add_argument("--worst", type=int, default=20, help="how many worst images to draw")
    args = ap.parse_args()
    sizes = [int(s) for s in args.imgsz.split(",")]

    model = YOLO(os.path.join(BASE_DIR, "yolo26n.pt"))
    files = sorted(f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(IMAGE_EXTS))
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []

    for imgsz in sizes:
        print(f"\nRunning YOLO at input size {imgsz} on {len(files)} images...")
        # Run once at a low confidence, then filter for each threshold below
        cache = {}
        for f in files:
            r = model.predict(os.path.join(IMAGES_DIR, f), conf=min(CONFS), imgsz=imgsz,
                              classes=[0], verbose=False)[0]
            preds = [(tuple(b.xywhn[0].tolist()), float(b.conf[0])) for b in r.boxes]
            cache[f] = (load_labels(os.path.splitext(f)[0]), preds)

        print("=" * 66)
        print(f"  INPUT SIZE {imgsz}")
        print("=" * 66)
        for conf in CONFS:
            for iou_thr in IOUS:
                tp = fp = gt_total = 0
                by_size = {name: [0, 0] for name, _, _ in SIZE_BINS}
                for f, (gts, preds) in cache.items():
                    p = [b for b, c in preds if c >= conf]
                    hit, n_matched = match(gts, p, iou_thr)
                    tp += len(hit)
                    fp += len(p) - n_matched
                    gt_total += len(gts)
                    for gi, g in enumerate(gts):
                        by_size[size_bin(g)][0] += 1
                        by_size[size_bin(g)][1] += gi in hit
                rec = 100 * tp / gt_total if gt_total else 0
                prec = 100 * tp / (tp + fp) if tp + fp else 0
                print(f"  conf {conf:.2f}  IoU>={iou_thr:.1f}   recall {rec:5.1f}%   precision {prec:5.1f}%"
                      f"   (caught {tp}/{gt_total}, false alarms {fp})")
                row = {"imgsz": imgsz, "conf": conf, "iou": iou_thr, "recall": round(rec, 1),
                       "precision": round(prec, 1), "caught": tp, "swimmers": gt_total, "false_alarms": fp}
                for name, (n, h) in by_size.items():
                    row[f"recall_{name.split()[0]}"] = round(100 * h / n, 1) if n else ""
                summary.append(row)
                if conf == 0.25 and iou_thr == 0.5:
                    size_line = "   ".join(f"{name}: {100 * h / n:.0f}% of {n}" for name, (n, h) in by_size.items() if n)
                    print(f"      recall by swimmer size -> {size_line}")

        # Pictures of the worst images (most missed swimmers), at the strict rule
        if imgsz == sizes[0]:
            scored = []
            for f, (gts, preds) in cache.items():
                p = [b for b, c in preds if c >= 0.25]
                hit, _ = match(gts, p, 0.5)
                scored.append((len(gts) - len(hit), f))
            vis_dir = os.path.join(OUT_DIR, f"worst_{imgsz}")
            os.makedirs(vis_dir, exist_ok=True)
            for missed, f in sorted(scored, reverse=True)[:args.worst]:
                gts, preds = cache[f]
                draw(os.path.join(IMAGES_DIR, f), gts, [(b, c) for b, c in preds if c >= 0.25],
                     os.path.join(vis_dir, f"missed{missed:02d}_{f}"))
            print(f"\n  Saved {min(args.worst, len(scored))} pictures to {vis_dir}")
            print("  GREEN = label (the answer key), RED = YOLO detection with its confidence")

    out_csv = os.path.join(OUT_DIR, "sanity_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nSummary table saved to {out_csv}")


if __name__ == "__main__":
    main()
