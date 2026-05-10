import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)  # 0 = laptop cam, 1 = DroidCam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        name = model.names[cls]
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1,y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("VisionAI", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()............................
cv2.destroyAllWindows()