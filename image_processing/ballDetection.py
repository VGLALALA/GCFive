import cv2
import numpy as np
import json
import os
from ultralytics import YOLO   # pip install ultralytics
# from .Convert_Canny import convert_to_canny   # keep if you still need it

MODEL_PATH  = "data/model/golfballv4.pt"
CLASS_ID    = 0   # change if your golf ball class id != 0

# ---------- YOLO STUFF ----------
model = YOLO(MODEL_PATH)

def detect_golfballs(image, conf=0.25, imgsz=640, display=True):
    """
    Run YOLO on an image (path or ndarray) and return a list of (x_center, y_center, r_pixels).
    r is approximated from the bbox as the average of half-width & half-height.
    """
    results = model.predict(source=image, conf=conf, imgsz=imgsz, verbose=False)
    if not results:
        return []

    boxes = results[0].boxes
    circle_data = []
    if boxes is None or len(boxes) == 0:
        return circle_data

    # xyxy → (x1,y1,x2,y2)
    for box in boxes:
        if hasattr(box, "cls") and int(box.cls.item()) != CLASS_ID:
            continue
        x_min, y_min, x_max, y_max = box.xyxy.cpu().numpy().astype(int).flatten()
        width, height = x_max - x_min, y_max - y_min
        radius = int((width + height) / 4)  # avg of (width/2, height/2)
        x_center, y_center = x_min + width // 2, y_min + height // 2
        circle_data.append((x_center, y_center, radius))

        if display:
            # Draw the bounding box and label on the image for debugging
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"Golf Ball: {int(box.cls.item())}"
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # sort left→right like your original code
    circle_data.sort(key=lambda c: c[0])
    print(f"Detected {len(circle_data)} golf ball(s).")

    if display:
        # Display the image with bounding boxes and labels
        cv2.imshow("Detected Golf Balls", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return circle_data
