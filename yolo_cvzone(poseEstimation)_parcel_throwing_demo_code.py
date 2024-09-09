import cv2
from ultralytics import YOLO
import numpy as np
import cvzone
import time

# Load YOLO model
model = YOLO("yolov8n-pose.pt")

# Path to the video file
video_path = r"E:\AI Ascending Software\AS AI Projects\AI PackageGuard\AI-PackageGuard\yolo\videos\WhatsApp Video 2024-08-30 at 21.13.02 (2).mp4"
# Open the video file
cap = cv2.VideoCapture(video_path)

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_width = 800
output_height = 450
out = cv2.VideoWriter('posees.mp4', fourcc, 30, (output_width, output_height))

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle

    return angle

# Initialize variables
last_capture_time = time.time()
capture_interval = 10  # seconds
zoom_factor = 2.0  # Factor to zoom in on the person
sitting_start_time = {}  # Track sitting start times for each person

while True:
    ret, frame = cap.read()
    
    # Break loop if no more frames
    if not ret:
        break

    # Resize frame
    try:
        frame = cv2.resize(frame, (output_width, output_height))
    except cv2.error as e:
        print(f"Error resizing frame: {e}")
        continue

    # Make predictions with the model
    results = model.predict(frame)

    # Get bounding box information in xyxy format
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    statuses = []

    # Get keypoints data
    keypoints_data = results[0].keypoints.data

    # Check if sitting and track time
    for i, keypoints in enumerate(keypoints_data):
        if keypoints.shape[0] > 0:
            angle = calculate_angle(keypoints[11][:2], keypoints[13][:2], keypoints[15][:2])
            status = 'Sitting' if angle < 110 else 'Standing'
            statuses.append(status)
            
            if status == 'Sitting':
                current_time = time.time()
                
                # Track start time of sitting
                if i not in sitting_start_time:
                    sitting_start_time[i] = current_time
                
                # Check if the person has been sitting for more than 10 seconds
                if current_time - sitting_start_time[i] >= capture_interval:
                    # Get the bounding box
                    x1, y1, x2, y2 = boxes[i]
                    
                    # Crop the region of interest (ROI) around the bounding box
                    roi = frame[y1:y2, x1:x2]
                    
                    # Calculate new dimensions for the zoomed image
                    height, width = roi.shape[:2]
                    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
                    
                    # Resize the ROI to zoom in
                    zoomed_roi = cv2.resize(roi, (new_width, new_height))
                    
                    # Save the zoomed-in frame as a photo
                    photo_filename = f'sitting_person_{time.strftime("%Y%m%d_%H%M%S")}.png'
                    cv2.imwrite(photo_filename, zoomed_roi)
                    print(f"Captured zoomed photo: {photo_filename}")
                    
                    # Remove from tracking after capturing photo
                    del sitting_start_time[i]
            else:
                # Reset sitting time if not sitting
                if i in sitting_start_time:
                    del sitting_start_time[i]
        else:
            statuses.append('Unknown')

    # Draw bounding boxes and statuses on the frame
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cvzone.putTextRect(
            frame, f"{statuses[i]}", (x1, y2 - 10),
            scale=1.5, thickness=2,
            colorT=(255, 255, 255), colorR=(255, 0, 255),
            font=cv2.FONT_HERSHEY_PLAIN,
            offset=10,
            border=1, colorB=(0, 255, 0)
        )

    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    detection = results[0].plot()
    cv2.imshow('YOLOv8 Pose Detection', detection)
    
    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()