import os
import csv
from ultralytics import YOLO

# 1. Load the YOLO model
model = YOLO("yolo26n.pt")

# 2. Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
SOURCES_CSV = os.path.join(DATASET_DIR, "sources.csv")

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
    with open(SOURCES_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0

    for row in rows:
        img_id = row["image_id"]
        fname = row["filename"]
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

    # Calculate final percentage scores
    precision = (total_tp / (total_tp + total_fp) * 100) if (total_tp + total_fp) > 0 else 0.0
    recall = (total_tp / total_gt * 100) if total_gt > 0 else 0.0

    return total_gt, total_tp, total_fp, total_fn, precision, recall


# 5. Run the evaluation!
print("=" * 60)
print("  EVALUATING YOLO26n ACROSS FULL 24-IMAGE BENCHMARK")
print("=" * 60)

for conf in [0.25, 0.15]:
    gt, tp, fp, fn, prec, rec = evaluate_dataset(conf)
    print(f"\nConfidence Threshold: {conf:.2f}")
    print(f"  Total Real Swimmers:  {gt}")
    print(f"  Swimmers Caught (TP): {tp}")
    print(f"  Swimmers Missed (FN): {fn}")
    print(f"  False Alarms    (FP): {fp}")
    print(f"  --> RECALL:           {rec:.1f}%")
    print(f"  --> PRECISION:        {prec:.1f}%")