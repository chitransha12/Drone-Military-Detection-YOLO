import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("C:/Users/HP/Downloads/Drone Military/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    results = model(frame[..., ::-1]) #YOLO inference 
    annotated_frame = results[0].plot()
    
    #Get image center
    h, w = frame.shape[:2]
    cx_img, cy_img = w // 2, h//2
    cv2.circle(annotated_frame, (cx_img, cy_img), 5, (0, 255, 255), -1)  #Yellow dot for image center
    
    for box in results[0].boxes:
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])   # bounding box coordinates
        conf = float(box.conf[0])               # confidence score
        cls = int(box.cls[0])                   # class id
        
        #Bounding box center
        cx_box = (xmin + xmax) // 2
        cy_box = (ymin + ymax) // 2
        
        #Draw box center
        cv2.circle(annotated_frame, (cx_box, cy_box), 5, (0, 0, 255), -1)    #red dot for box center
        
        #Distance Calculation
        dx = cx_box - cx_img
        dy = cy_box - cy_img
        distance = (dx**2 + dy**2)**0.5
        
        #to show result on bounding box
        text = f"Cls:{cls} Conf:{conf:.2f} Dist:{int(distance)}px dx:{dx} dy:{dy}"
        cv2.putText(annotated_frame, text, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Print values in terminal
        print(
            f"Class: {cls}, Conf: {conf:.2f}, BBoxCenter=({cx_box}, {cy_box}), "
            f"ImgCenter=({cx_img}, {cy_img}), dx={dx}, dy={dy}, Dist={distance:.2f}px")
        
    cv2.imshow("YOLO Webcam", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()