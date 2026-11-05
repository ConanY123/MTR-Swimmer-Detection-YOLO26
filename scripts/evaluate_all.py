import os
import csv
from ultralytics import YOLO

# 1. Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
RESULTS_DIR = os.path.join(DATASET_DIR, "eval_results")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 2. Load the YOLO model (found next to this repo, whatever folder you run from)
model = YOLO(os.path.join(BASE_DIR, "yolo26n.pt"))

# 3. IoU Function
def compute_iou(box1, box2):
    x1_min, y1_min = box1[0] - box1[2] / 2.0, box1[1] - box1[3] / 2.0
    x1_max, y1_max = box1[0] + box1[2] / 2.0, box1[1] + box1[3] / 2.0
    x2_min, y2_min = box2[0] - box2[2] / 2.0, box2[1] - box2[3] / 2.0
    x2_max, y2_max = box2[0] + box2[2] / 2.0, box2[1] + box2[3] / 2.0

    inter_w = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = inter_w * inter_h

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


# 4. Evaluation Function for any confidence threshold
def evaluate_dataset(conf_threshold):
    # Evaluate every image in dataset/images (not just the ones in sources.csv),
    # so images imported from Roboflow are included too.
    image_files = sorted(
        f for f in os.listdir(IMAGES_DIR)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    per_image = []

    for fname in image_files:
        img_id = os.path.splitext(fname)[0]
        img_path = os.path.join(IMAGES_DIR, fname)
        lbl_path = os.path.join(LABELS_DIR, f"{img_id}.txt")

        # Load ground truth boxes for this image
        gt_boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        gt_boxes.append([float(x) for x in parts[1:5]])

        total_gt += len(gt_boxes)

        # Run model inference
        results = model.predict(img_path, conf=conf_threshold, verbose=False)
        pred_boxes = []
        for b in results[0].boxes:
            if int(b.cls[0]) == 0:  # Class 0: person
                pred_boxes.append(b.xywhn[0].tolist())

        # Match predictions to ground truth
        matched_gt = set()
        img_tp = 0
        img_fp = 0

        for p in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1

            # Find which ground truth swimmer this prediction overlaps most
            for g_idx, g_box in enumerate(gt_boxes):
                if g_idx not in matched_gt:
                    iou = compute_iou(p, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

            # If overlap >= 50%, it's a True Positive!
            if best_iou >= 0.5 and best_gt_idx != -1:
                matched_gt.add(best_gt_idx)
                img_tp += 1
            else:
                img_fp += 1  # False alarm (buoy, wave, or duplicate)

        img_fn = len(gt_boxes) - len(matched_gt)

        total_tp += img_tp
        total_fp += img_fp
        total_fn += img_fn
        per_image.append({
            "image": fname,
            "source": "roboflow" if img_id.startswith("rf_") else "conan_open_water",
            "real_swimmers": len(gt_boxes),
            "caught_tp": img_tp,
            "missed_fn": img_fn,
            "false_alarms_fp": img_fp,
        })

    # Save a per-image breakdown, so we can see which images the misses
    # and false alarms come from
    if not per_image:
        raise SystemExit(f"No images found in {IMAGES_DIR}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, f"per_image_conf_{conf_threshold:.2f}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)

    # Calculate final percentage scores
    precision = (total_tp / (total_tp + total_fp) * 100) if (total_tp + total_fp) > 0 else 0.0
    recall = (total_tp / total_gt * 100) if total_gt > 0 else 0.0

    return total_gt, total_tp, total_fp, total_fn, precision, recall, len(image_files), out_csv


# 5. Run the evaluation!
print("=" * 60)
print("  EVALUATING YOLO26n ON ALL IMAGES IN dataset/images")
print("=" * 60)

for conf in [0.25, 0.15]:
    gt, tp, fp, fn, prec, rec, n_images, out_csv = evaluate_dataset(conf)
    print(f"\nConfidence Threshold: {conf:.2f}  ({n_images} images)")
    print(f"  Total Real Swimmers:  {gt}")
    print(f"  Swimmers Caught (TP): {tp}")
    print(f"  Swimmers Missed (FN): {fn}")
    print(f"  False Alarms    (FP): {fp}")
    print(f"  --> RECALL:           {rec:.1f}%")
    print(f"  --> PRECISION:        {prec:.1f}%")
    print(f"  Per-image breakdown:  {out_csv}")