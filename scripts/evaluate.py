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
    print(f" Class:   {cls_name} (ID: {cls_id})")
    print(f" Confidence: {conf:.2f}")
    print(f" Box:    {[round(x,4) for x in xywh]}")
    print("-" * 35)


