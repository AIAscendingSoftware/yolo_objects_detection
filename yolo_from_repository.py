import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # ultralytics/yolov5-path,yolov5x-model 'yolov5s' for small model ou can replace 'yolov5s' with 'yolov5m' or 'yolov5l' for different versions.

# Open video feed
video_path = r"D:\downloads from chrome\WhatsApp Video 2024-08-30 at 21.13.04 (3).mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Draw results on frame
    frame = results.render()[0]

    # Display the frame
    cv2.imshow('Live Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

