
import cv2
import torch
from ultralytics import YOLO

def load_model(model_type):
    """Load and return the specified YOLO model."""
    if model_type == 'v5':
        return torch.hub.load('ultralytics/yolov5', 'yolov5x')
    elif model_type == 'v8':
        return YOLO('yolov8x.pt')

def process_video(video_path, model_type):
    """Process video using the specified YOLO model."""
    model = load_model(model_type)
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)
        
        # Draw results on frame
        if model_type == 'v5':
            frame = results.render()[0]
        else:
            annotated_frame = results[0].plot()
        
        # Display the frame
        cv2.imshow(f'YOLO{model_type} Detection', frame if model_type == 'v5' else annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage:
# process_video(r"D:\downloads from chrome\WhatsApp Video 2024-08-30 at 21.13.04 (2).mp4", 'v5')
process_video(r"D:\downloads from chrome\WhatsApp Video 2024-08-30 at 21.13.04 (2).mp4", 'v8')
