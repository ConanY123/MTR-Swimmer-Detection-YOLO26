#!/usr/bin/env python3
"""
Fine-tunes YOLO26n on the swimmer training set built by build_train_set.py.

It starts from the normal yolo26n.pt (which already knows what a person
looks like) and adjusts it for swimmers, instead of starting from scratch.

Version 2 fine-tunes gently, because version 1 "forgot" large, close people
(recall on the eval set fell from 23.8% to 4.9%):
  * It keeps all 80 COCO classes, so the part of the model that recognises
    "person" is kept instead of being rebuilt from our narrow data. Our
    labels use class 0, which is "person" in COCO.
  * It locks (freezes) the first 10 layers, which recognise basic shapes.
  * It uses a lower learning rate and fewer epochs, so it adjusts instead of
    relearning.

Usage (from the repo root, ideally on a GPU, e.g. Google Colab):
    python scripts/train_swimmer.py                       # v2 settings
    python scripts/train_swimmer.py --epochs 30 --freeze 10 --lr0 0.002

The trained model is saved to models/yolo26n_swimmer_v2.pt (change with --name).
Then test it on the eval set:
    python scripts/evaluate_all.py --model models/yolo26n_swimmer_v2.pt
"""

import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(BASE_DIR / "train_data" / "swimmers.yaml"))
    p.add_argument("--name", default="yolo26n_swimmer_v2")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--freeze", type=int, default=10, help="number of early layers to lock (0 = none)")
    p.add_argument("--lr0", type=float, default=0.002, help="starting learning rate")
    args = p.parse_args()

    model = YOLO(str(BASE_DIR / "yolo26n.pt"))

    # Use the model's own 80 class names, so its whole detection head is kept
    data = yaml.safe_load(Path(args.data).read_text())
    data["names"] = dict(model.names)
    data_80 = Path(args.data).with_name("swimmers_coco80.yaml")
    data_80.write_text(yaml.safe_dump(data))

    model.train(
        data=str(data_80),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        optimizer="SGD",      # "auto" would ignore lr0
        lr0=args.lr0,
        patience=8,           # stop early if it stops improving
        project=str(BASE_DIR / "runs"),
        name=args.name,
        exist_ok=True,
        seed=0,
    )

    best = BASE_DIR / "runs" / args.name / "weights" / "best.pt"
    out = BASE_DIR / "models" / f"{args.name}.pt"
    out.parent.mkdir(exist_ok=True)
    shutil.copy(best, out)
    print(f"\nSaved the fine-tuned model to {out}")


if __name__ == "__main__":
    main()
