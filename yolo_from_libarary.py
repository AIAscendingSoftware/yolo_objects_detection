
import cv2
from ultralytics import YOLO
model = YOLO('yolov8x.pt')

# Open video feed
video_path = r"D:\downloads from chrome\WhatsApp Video 2024-08-30 at 21.13.04 (3).mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Perform object detection
    results = model(frame)
    
    # results is a list of results, usually we are interested in the first item
    result = results[0]

    # Draw results on frame
    annotated_frame = result.plot()  # Use result.plot() to draw bounding boxes

    # Display the frame
    cv2.imshow('Live Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()