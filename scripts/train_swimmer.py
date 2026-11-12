#!/usr/bin/env python3
"""
Fine-tunes YOLO26n on the swimmer training set built by build_train_set.py.

It starts from the normal yolo26n.pt (which already knows what a person
looks like) and teaches it swimmers, instead of starting from scratch.
Class 0 stays "person", so evaluate_all.py works on the new model unchanged.

Usage (from the repo root, ideally on a GPU, e.g. Google Colab):
    python scripts/train_swimmer.py                 # 40 epochs
    python scripts/train_swimmer.py --epochs 60

The trained model is saved to models/yolo26n_swimmer.pt.
Then test it on the eval set:
    python scripts/evaluate_all.py --model models/yolo26n_swimmer.pt
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(BASE_DIR / "train_data" / "swimmers.yaml"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()

    model = YOLO(str(BASE_DIR / "yolo26n.pt"))
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=10,          # stop early if it stops improving
        project=str(BASE_DIR / "runs"),
        name="swimmer",
        exist_ok=True,
        seed=0,
    )

    best = BASE_DIR / "runs" / "swimmer" / "weights" / "best.pt"
    out = BASE_DIR / "models" / "yolo26n_swimmer.pt"
    out.parent.mkdir(exist_ok=True)
    shutil.copy(best, out)
    print(f"\nSaved the fine-tuned model to {out}")


if __name__ == "__main__":
    main()
