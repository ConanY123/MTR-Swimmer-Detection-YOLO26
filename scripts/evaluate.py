from ultralytics import YOLO

"""
Running the data for one image
"""
# Load yolo model
model = YOLO("yolo26n.pt")
# Test image path
image_path = "dataset/images/swimmer_eval_001.jpg"

# Run prediction -> List containing results objects
results = model.predict(image_path, conf=0.10, verbose=False)

# box object
boxes = results[0].boxes


print(f"Total objects detected:{len(boxes)}\n")

for i, box in enumerate(boxes):
    cls_id = int(box.cls[0])        # class id 0, 1, etc
    cls_name = model.names[cls_id]  # Name for the class ie 'person'
    conf=float(box.conf[0])         # confidence 0.0-1.0    
    xywh=box.xywhn[0].tolist();     # Normalized [center_x, center_y, width, height], all relative to full image size

    print(f"Detection #{i+1}:")
    print(f"  Class:   {cls_name} (ID: {cls_id})")
    print(f"  Confidence: {conf:.2f}")
    print(f"  Box:    {[round(x,4) for x in xywh]}")
    print("-" * 35)

"""
Loading the Ground Truth Labels (answer key)
"""
# Answer key path
label_path = "dataset/labels/swimmer_eval_001.txt"
# Empty list
gt_boxes = []

# Populating the ground truth boxes list
with open(label_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        if parts:
            # parts[0] is the class ID (0)
            # parts[1:5] are [center_x, center_y, width, height]
            box = [float(x) for x in parts[1:5]]
            gt_boxes.append(box)

# gt_boxes is now a list of lists containing 4 coordinates
print(f"\nGround Truth Swimmers: {len(gt_boxes)}")
for i, g in enumerate(gt_boxes):
    print(f" GT #{i+1} Box: {[round(x,4) for x in g]}")

"""
Intersection over Union and evaluation
"""
# Function for calculating IoU
def compute_iou(box1, box2):
    # Convert normalized [center_x, center_y, width, height] into [xmin, ymin, xmax, ymax]
    x1_min, y1_min = box1[0] - box1[2] / 2.0, box1[1] - box1[3] / 2.0
    x1_max, y1_max = box1[0] + box1[2] / 2.0, box1[1] + box1[3] / 2.0
    x2_min, y2_min = box2[0] - box2[2] / 2.0, box2[1] - box2[3] / 2.0
    x2_max, y2_max = box2[0] + box2[2] / 2.0, box2[1] + box2[3] / 2.0

    # Calculate intersection rectangle width and height
    inter_w = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))

    # Area
    intersection = inter_w * inter_h

    # Calculate union box area
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection

    # Return IoU or 0 if no intersection
    return intersection/union if union > 0 else 0.0

# Comparing each detection against the GT swimmers
print("\n" + "=" * 45)
print("  Comparing Model Detections vs Ground Truth")
print("=" * 45)

gt_swimmer = gt_boxes[0]
for i, box in enumerate(boxes):
    pred_box = box.xywhn[0].tolist()
    conf = float(box.conf[0])
    iou = compute_iou(pred_box, gt_swimmer)

    # Check if it counts as a hit (IoU >= 0.5)
    is_hit = iou >= 0.5
    status = "HIT (Matches Swimmer!)" if is_hit else "MISS (Not the swimmer)"

    print(f"Detection #{i+1} (Conf: {conf:.2f}):")
    print(f"  Overlap (IoU): {iou:.4f} ({iou * 100:.1f}%)")
    print(f"  Result:        {status}")
    print("-" * 45)


# Tally TP, FP, and FN for the image
tp = 0
fp = 0
matched_gt = set()

# Check each deteciton(highest confidence first)
for i, box in enumerate(boxes):
    pred_box = box.xywhn[0].tolist()
    conf = float(box.conf[0])

    # Calculate IoU against ground truth swimmmer
    iou = compute_iou(pred_box, gt_swimmer)
    
    # Only a TP if IoU >= 0.5 AND the swimmmer hasnt been detected by another instance
    if iou>=0.5 and 0 not in matched_gt:
        tp += 1
        matched_gt.add(0) # Mark ground truth swimmer #0 as claimed
        print(f"Detection #{i+1}: TRUE POSITIVE (Real swimmer found!)")
    else:
        fp += 1
        reason = "Overlap <50%" if iou < 0.5 else "Duplicate box on same swimmer"
        print(f"Detection #{i+1}: FALSE POSITIVE ({reason})")

# Any ground truth swimmers not in matched_gt is a Miss
fn = len(gt_boxes) - len(matched_gt)

print("\n" + "=" * 35)
print(f"FINAL SCORE FOR IMAGE 1:")
print(f"  True Positives  (TP): {tp} (Hit)")
print(f"  False Positives (FP): {fp} (False alarms: bag + duplicate)")
print(f"  False Negatives (FN): {fn} (Misses)")
print("=" * 35)