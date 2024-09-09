# import cv2
# import numpy as np
# from ultralytics import YOLO

# # Initialize YOLO model
# model = YOLO('yolov8x.pt')

# # Initialize OpenPose
# net = cv2.dnn.readNetFromCaffe("E:\AI Ascending Software\AS AI Projects\AI PackageGuard\AI-PackageGuard\yolo\openpose dependencies\openpose\models\pose\mpi\pose_deploy_linevec.prototxt", "E:\AI Ascending Software\AS AI Projects\AI PackageGuard\AI-PackageGuard\yolo\openpose dependencies\pose_iter_440000.caffemodel")

# # COCO Output Format
# BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
#                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
#                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
#                "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

# POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
#                ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
#                ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
#                ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
#                ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

# # Open video feed
# video_path = r"D:\downloads from chrome\WhatsApp Video 2024-08-30 at 21.13.04 (3).mp4"
# cap = cv2.VideoCapture(video_path)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Perform YOLO object detection
#     results = model(frame)
    
#     # Draw YOLO results on frame
#     annotated_frame = results[0].plot()
    
#     # Prepare the frame for OpenPose
#     frameWidth = frame.shape[1]
#     frameHeight = frame.shape[0]
#     inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
#     print('inpBlob:',inpBlob)
#     net.setInput(inpBlob)
#     output = net.forward()

#     H = output.shape[2]
#     W = output.shape[3]

#     # Empty list to store the detected keypoints
#     points = []

#     for i in range(len(BODY_PARTS)):
#         # Slice heatmap of corresponding body part.
#         probMap = output[0, i, :, :]

#         # Find global maxima of the probMap.
#         minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

#         # Scale the point to fit on the original image
#         x = (frameWidth * point[0]) / W
#         y = (frameHeight * point[1]) / H

#         if prob > 0.1:
#             cv2.circle(annotated_frame, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
#             cv2.putText(annotated_frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
#             points.append((int(x), int(y)))
#         else:
#             points.append(None)

#     # Draw Skeleton
#     for pair in POSE_PAIRS:
#         partA = pair[0]
#         partB = pair[1]

#         idA = BODY_PARTS[partA]
#         idB = BODY_PARTS[partB]

#         if points[idA] and points[idB]:
#             cv2.line(annotated_frame, points[idA], points[idB], (0, 255, 0), 3)

#     # Display the frame
#     cv2.imshow('Live Detection with Pose Estimation', annotated_frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('yolov8x.pt')

# Initialize OpenPose
net = cv2.dnn.readNetFromCaffe(
    "E:\AI Ascending Software\AS AI Projects\AI PackageGuard\AI-PackageGuard\yolo\openpose dependencies\openpose\models\pose\mpi\pose_deploy_linevec.prototxt",
    "E:\AI Ascending Software\AS AI Projects\AI PackageGuard\AI-PackageGuard\yolo\openpose dependencies\pose_iter_440000.caffemodel"
)

# COCO Output Format
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

# Open video feed
video_path = r"D:\downloads from chrome\WhatsApp Video 2024-08-30 at 21.13.04 (3).mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform YOLO object detection
    results = model(frame)
    
    # Draw YOLO results on frame
    annotated_frame = results[0].plot()
    
    # Prepare the frame for OpenPose
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
    
    print('Frame shape:', frame.shape)
    print('inpBlob shape:', inpBlob.shape)
    
    net.setInput(inpBlob)
    
    try:
        output = net.forward()
        print('Output shape:', output.shape)
    except cv2.error as e:
        print('OpenCV Error:', e)
        break

    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []

    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > 0.1:
            cv2.circle(annotated_frame, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(annotated_frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
        else:
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        idA = BODY_PARTS[partA]
        idB = BODY_PARTS[partB]

        if points[idA] and points[idB]:
            cv2.line(annotated_frame, points[idA], points[idB], (0, 255, 0), 3)

    # Display the frame
    cv2.imshow('Live Detection with Pose Estimation', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
