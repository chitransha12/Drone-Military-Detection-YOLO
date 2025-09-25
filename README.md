#🛰️ Drone Military Detection using YOLOv8

📌 Overview
This project implements a YOLOv8-based Drone Detection System trained on a custom Drone Military dataset (Roboflow/Kaggle).
It detects and tracks drones in images, test datasets, and live webcam streams, while also calculating bounding box distance from image center for alignment and tracking.

Applications include defense, surveillance, aerospace, and autonomous systems.

⚙️ Features
✅ Train YOLOv8 on custom Drone dataset using Kaggle
✅ Run inference on test images with predictions saved automatically
✅ Real-time webcam detection using OpenCV
✅ Calculate object center, dx, dy, and distance from image center
✅ Overlay results with confidence, class, and pixel distance

📂 Dataset
Dataset: Drone Military (Roboflow)
Format: YOLOv8-compatible YAML with train/val/test splits
Example path: /kaggle/input/drone-military/

🚀 Training on Kaggle
!ls /kaggle/input/
!ls /kaggle/input/drone-military/
!cat /kaggle/input/drone-military/data.yaml  

# Install YOLOv8
!pip install ultralytics -q

# Train the Drone Military Model
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

model.train(
    data="/kaggle/input/drone-military/data.yaml",
    epochs=20,
    imgsz=640
)

🔍 Testing on Images
from ultralytics import YOLO

# Load trained model
model = YOLO("/kaggle/working/runs/detect/train/weights/best.pt")

# Run predictions on test images
results = model.predict(
    source="/kaggle/input/drone-military/test/images",
    save=True
)

print("✅ Done! Check runs/detect/predict folder.")

🎥 Real-Time Webcam Inference
import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("C:/Users/HP/Downloads/Drone Military/best.pt")  # update path if needed

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model(frame[..., ::-1])
    annotated_frame = results[0].plot()

    # Image center
    h, w = frame.shape[:2]
    cx_img, cy_img = w // 2, h // 2
    cv2.circle(annotated_frame, (cx_img, cy_img), 5, (0, 255, 255), -1)

    # Loop through detections
    for box in results[0].boxes:
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        # Bounding box center
        cx_box = (xmin + xmax) // 2
        cy_box = (ymin + ymax) // 2
        cv2.circle(annotated_frame, (cx_box, cy_box), 5, (0, 0, 255), -1)

        # Distance calculation
        dx, dy = cx_box - cx_img, cy_box - cy_img
        distance = (dx**2 + dy**2)**0.5

        # Display info
        text = f"Cls:{cls} Conf:{conf:.2f} Dist:{int(distance)}px dx:{dx} dy:{dy}"
        cv2.putText(annotated_frame, text, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Print in terminal
        print(f"Class:{cls}, Conf:{conf:.2f}, "
              f"BoxCenter=({cx_box},{cy_box}), ImgCenter=({cx_img},{cy_img}), "
              f"dx={dx}, dy={dy}, Dist={distance:.2f}px")

    cv2.imshow("YOLO Webcam", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

📊 Results
Detections stored in: runs/detect/predict/
Console logs include:
Class ID
Confidence
Bounding Box center
dx, dy, and distance (in pixels)

🛠️ Tech Stack
Python
YOLOv8 (Ultralytics)
OpenCV
Kaggle

📌 Future Scope
Integrate drone tracking + autopilot control
Deploy on edge devices (Jetson, Raspberry Pi, UAVs)
Extend to multi-class detection (soldiers, vehicles, drones, etc.)
