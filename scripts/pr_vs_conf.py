#!/usr/bin/env python3
"""
Precision and recall against the confidence threshold, for one or more models,
on the eval set (dataset/images + dataset/labels).

The confidence threshold is how sure YOLO must be before a box counts.
A low threshold finds more swimmers (higher recall) but adds false alarms
(lower precision); a high one does the opposite. This chart shows that
trade-off for each model, so we can pick a threshold and compare models.

Usage (from the repo root):
    python scripts/pr_vs_conf.py                                   # normal yolo26n
    python scripts/pr_vs_conf.py --models yolo26n.pt models/yolo26n_swimmer_v2.pt

Saves to dataset/eval_results/:
    pr_vs_conf.png   the chart (recall and precision vs confidence, one line per model)
    pr_vs_conf.csv   the numbers behind it
It also prints recall by swimmer size at confidence 0.25 for each model.
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sanity_check import SIZE_BINS, load_labels, match, size_bin  # same matching as the sanity check

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "images")
OUT_DIR = os.path.join(BASE_DIR, "dataset", "eval_results")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 ... 0.95
IOU_THR = 0.5


def run_model(path, files):
    """Runs the model once at a very low threshold; returns {stem: [(conf, box), ...]}."""
    model = YOLO(path)
    preds = {}
    for f in files:
        r = model.predict(os.path.join(IMAGES_DIR, f), conf=0.01, classes=[0], verbose=False)[0]
        preds[os.path.splitext(f)[0]] = [(float(b.conf[0]), tuple(b.xywhn[0].tolist())) for b in r.boxes]
    return preds


def scores(preds, gts, thr):
    tp = fp = n_gt = 0
    by_size = {name: [0, 0] for name, _, _ in SIZE_BINS}  # name -> [found, total]
    for stem, g in gts.items():
        p = [box for c, box in preds[stem] if c >= thr]
        found, n_matched = match(g, p, IOU_THR)
        tp += n_matched
        fp += len(p) - n_matched
        n_gt += len(g)
        for i, box in enumerate(g):
            by_size[size_bin(box)][1] += 1
            by_size[size_bin(box)][0] += i in found
    recall = tp / n_gt if n_gt else 0.0
    precision = tp / (tp + fp) if tp + fp else float("nan")  # no boxes at all: leave a gap
    return recall, precision, by_size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=[os.path.join(BASE_DIR, "yolo26n.pt")])
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(IMAGE_EXTS))
    gts = {os.path.splitext(f)[0]: load_labels(os.path.splitext(f)[0]) for f in files}
    print(f"{len(files)} images, {sum(map(len, gts.values()))} swimmers, IoU >= {IOU_THR}\n")

    rows, curves = [], {}
    for m in args.models:
        name = os.path.splitext(os.path.basename(m))[0]
        preds = run_model(m, files)
        rec, prec = [], []
        for t in THRESHOLDS:
            r, p, _ = scores(preds, gts, t)
            rec.append(r * 100)
            prec.append(p * 100)
            rows.append({"model": name, "confidence": t, "recall_pct": round(r * 100, 1),
                         "precision_pct": "" if p != p else round(p * 100, 1)})
        curves[name] = (rec, prec)
        _, _, by_size = scores(preds, gts, 0.25)
        sizes = "   ".join(f"{k}: {f}/{n} ({100 * f / n:.0f}%)" if n else f"{k}: -"
                           for k, (f, n) in by_size.items())
        print(f"{name}  recall by swimmer size at conf 0.25 ->  {sizes}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "pr_vs_conf.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Two panels side by side, same x axis, one colour per model
    fig, (ax_r, ax_p) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    colors = ["#2f6db5", "#d1603d", "#3a9a5b", "#8a5cc2", "#b0891f"]
    for (name, (rec, prec)), c in zip(curves.items(), colors):
        ax_r.plot(THRESHOLDS, rec, color=c, lw=2, label=name)
        ax_p.plot(THRESHOLDS, prec, color=c, lw=2, label=name)
    for ax, title in ((ax_r, "Recall: share of swimmers found"),
                      (ax_p, "Precision: share of boxes that are real swimmers")):
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel("%")
        ax.set_ylim(0, 100)
        ax.set_xlim(0.05, 0.95)
        ax.axvline(0.25, color="#999", lw=1, ls="--")
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax_r.text(0.26, 96, "0.25 (default)", fontsize=8, color="#666", va="top")
    ax_r.legend(frameon=False, fontsize=9)
    fig.suptitle(f"YOLO26n on the swimmer eval set ({len(files)} images, IoU ≥ {IOU_THR})",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "pr_vs_conf.png")
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved {out_png} and pr_vs_conf.csv")


if __name__ == "__main__":
    main()
