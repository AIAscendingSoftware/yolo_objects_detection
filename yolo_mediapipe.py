import cv2
import mediapipe as mp
from ultralytics import YOLO

# Initialize YOLOv8 model
model = YOLO('yolov8x.pt')

# Initialize Mediapipe Pose for body landmarks
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Open video feed
video_path = r"D:\downloads from chrome\WhatsApp Video 2024-08-30 at 21.13.04 (3).mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform object detection with YOLOv8
    results = model(frame)
    result = results[0]  # YOLO result for the current frame
    
    # Extract bounding boxes for detected people
    for box in result.boxes:
        if int(box.cls[0]) == 0:  # Check if the detected class is 'person'
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop the detected person region for pose estimation
            person_frame = frame[y1:y2, x1:x2]
            person_frame_rgb = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)

            # Perform pose estimation on the cropped person
            pose_results = pose.process(person_frame_rgb)

            # Draw pose landmarks if found
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    person_frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
            
            # Place the modified person_frame back in the original frame
            frame[y1:y2, x1:x2] = person_frame

    # Draw YOLOv8 bounding boxes on the frame
    annotated_frame = result.plot()

    # Display the combined frame (YOLO + Pose Estimation)
    cv2.imshow('YOLOv8 + Pose Estimation', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
