import cv2
from ultralytics import YOLO
import numpy as np
from itertools import combinations
import joblib
from collections import deque
import time

# Load YOLOv8n-pose model
print("Loading YOLO model...")
model = YOLO('models/yolov8n-pose.pt')

# Load trained classifier and scaler
print("Loading attention classifier...")
classifier = joblib.load("./models/attention_classifier.pkl")
scaler = joblib.load("./models/scaler.pkl")

# Upper body keypoint indices (COCO format)
upper_body_indices = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10
}

keypoint_names = list(upper_body_indices.keys())
keypoint_pairs = list(combinations(keypoint_names, 2))

# Smoothing buffer for predictions
prediction_buffer = deque(maxlen=10)  # Store last 10 predictions

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def extract_features(keypoints_data):
    """Extract distance features from keypoints for a single person"""
    # Extract upper body keypoints
    upper_body_kps = {}
    
    for name, idx in upper_body_indices.items():
        x, y, conf = keypoints_data[idx]
        upper_body_kps[name] = (float(x), float(y), float(conf))
    
    # Calculate all distances
    features = []
    for pair in keypoint_pairs:
        p1 = upper_body_kps[pair[0]]
        p2 = upper_body_kps[pair[1]]
        
        # Only calculate distance if both points are confident
        if p1[2] > 0.5 and p2[2] > 0.5:
            dist = calculate_distance(p1[:2], p2[:2])
        else:
            dist = 0  # Use 0 instead of -1 for consistency with training
        
        features.append(float(dist))
    
    return features

def get_smoothed_prediction(current_pred):
    """Smooth predictions using a rolling buffer"""
    prediction_buffer.append(current_pred)
    return np.mean(prediction_buffer) > 0.5

# Open webcam
print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

print("\n" + "="*60)
print("Real-Time Attention Detection System")
print("="*60)
print("Press 'q' to quit")
print("Press 'r' to reset prediction buffer")
print("="*60 + "\n")

frame_count = 0
fps_time = time.time()
fps = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Calculate FPS
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - fps_time)
            fps_time = time.time()
        
        # Run YOLOv8 pose detection
        results = model(frame, verbose=False)
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        # Extract keypoints and make predictions
        if results[0].keypoints is not None:
            all_keypoints = results[0].keypoints.data.cpu().numpy()
            
            # Process each detected person
            for person_id, full_kps in enumerate(all_keypoints):
                try:
                    # Extract features
                    features = extract_features(full_kps)
                    
                    # Prepare for prediction
                    features_array = np.array(features).reshape(1, -1)
                    features_scaled = scaler.transform(features_array)
                    
                    # Make prediction
                    prediction = classifier.predict(features_scaled)[0]
                    probability = classifier.predict_proba(features_scaled)[0]
                    
                    # Get smoothed prediction
                    smoothed_pred = get_smoothed_prediction(prediction)
                    
                    # Determine label and color
                    if smoothed_pred:
                        label = "ATTENTION"
                        color = (0, 255, 0)  # Green
                        conf = probability[1]
                    else:
                        label = "INATTENTION"
                        color = (0, 0, 255)  # Red
                        conf = probability[0]
                    
                    # Get nose position for text placement
                    nose_idx = upper_body_indices['nose']
                    nose_x, nose_y, nose_conf = full_kps[nose_idx]
                    
                    if nose_conf > 0.3:
                        # Draw background rectangle for text
                        text = f"Person {person_id}: {label}"
                        conf_text = f"Confidence: {conf:.2f}"
                        
                        text_x = int(nose_x) - 80
                        text_y = int(nose_y) - 60
                        
                        # Draw semi-transparent background
                        overlay = annotated_frame.copy()
                        cv2.rectangle(overlay, 
                                    (text_x - 10, text_y - 35),
                                    (text_x + 250, text_y + 30),
                                    color, -1)
                        cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
                        
                        # Draw text
                        cv2.putText(annotated_frame, text,
                                  (text_x, text_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(annotated_frame, conf_text,
                                  (text_x, text_y + 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                except Exception as e:
                    print(f"Error processing person {person_id}: {e}")
        
        # Display info overlay
        info_y = 30
        cv2.putText(annotated_frame, f'FPS: {fps:.1f}', 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f'Frame: {frame_count}', 
                   (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, 'Press Q to quit | R to reset', 
                   (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Attention Detection System', annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            prediction_buffer.clear()
            print("Prediction buffer reset")

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\n" + "="*60)
    print("System stopped")
    print(f"Total frames processed: {frame_count}")
    print("="*60)