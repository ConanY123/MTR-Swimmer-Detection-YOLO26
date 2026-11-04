#!/usr/bin/env python3
"""
"lables" folder should have .txt files with box coordinates of where we know the swimmers are (dataset)
This basically creates images in dataset/visual_verification/ that has the boxes so we can look at it
"""

import os
import csv
from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
VERIFY_DIR = os.path.join(DATASET_DIR, "visual_verification")
SOURCES_CSV = os.path.join(DATASET_DIR, "sources.csv")

os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(VERIFY_DIR, exist_ok=True)

def main():
    print("=== Rendering Ground Truth Visual Verification Gallery ===")
    
    if not os.path.exists(SOURCES_CSV):
        print(f"Error: {SOURCES_CSV} not found!")
        return

    with open(SOURCES_CSV, 'r', encoding='utf-8') as f:
        sources = list(csv.DictReader(f))
        
    print(f"Auditing and rendering {len(sources)} images...")
    
    total_swimmers = 0
    positive_count = 0
    negative_count = 0
    
    for row in sources:
        img_id = row['image_id']
        fname = row['filename']
        is_swimmer = int(row['is_swimmer'])
        
        img_path = os.path.join(IMAGES_DIR, fname)
        label_path = os.path.join(LABELS_DIR, f"{img_id}.txt")
        verify_path = os.path.join(VERIFY_DIR, f"{img_id}_verified.jpg")
        
        if not os.path.exists(img_path):
            continue
            
        with Image.open(img_path) as im:
            img_w, img_h = im.size
            vis_img = im.convert("RGB")
            
        draw = ImageDraw.Draw(vis_img)
        
        # Read ground truth boxes from label file
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                        boxes.append([xc, yc, bw, bh])
                        
        if is_swimmer == 0 or len(boxes) == 0:
            negative_count += 1
            # Draw negative watermark
            draw.rectangle([(10, 10), (450, 45)], fill=(40, 40, 40))
            draw.text((20, 18), f"[{img_id}] NEGATIVE CONTROL: 0 Swimmers", fill=(0, 255, 255))
        else:
            positive_count += 1
            total_swimmers += len(boxes)
            
            # Draw green bounding boxes for visual verification
            for idx, box in enumerate(boxes):
                xc, yc, bw, bh = box
                xmin = (xc - bw / 2.0) * img_w
                ymin = (yc - bh / 2.0) * img_h
                xmax = (xc + bw / 2.0) * img_w
                ymax = (yc + bh / 2.0) * img_h
                
                draw.rectangle([xmin, ymin, xmax, ymax], outline="#00FF00", width=3)
                label_txt = f"GT Swimmer #{idx+1}"
                draw.rectangle([xmin, max(0, ymin - 18), xmin + 120, max(0, ymin)], fill="#00FF00")
                draw.text((xmin + 4, max(0, ymin - 16)), label_txt, fill="black")
                
            # Header info
            draw.rectangle([(10, 10), (450, 45)], fill=(0, 100, 0))
            draw.text((20, 18), f"[{img_id}] SWIMMER: {len(boxes)} in water", fill="white")
            
        vis_img.save(verify_path, "JPEG", quality=90)
        
    print(f"\n[DONE] Verification gallery generated:")
    print(f"  - Positive images: {positive_count} ({total_swimmers} ground-truth swimmers)")
    print(f"  - Negative images: {negative_count} (empty text files)")
    print(f"  - Label files:      {LABELS_DIR}")
    print(f"  - Visual gallery:   {VERIFY_DIR}")

if __name__ == '__main__':
    main()
