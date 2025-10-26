import cv2
from ultralytics import YOLO
import csv
from datetime import datetime
import numpy as np
from itertools import combinations
import os

# Load YOLOv8n-pose model
model = YOLO('models/yolov8n-pose.pt')

# Open webcam
cap = cv2.VideoCapture(0)

# Dataset base directory
data_dir = "./dataset"
os.makedirs(data_dir, exist_ok=True)

# Category
category = input("Enter Category: ").strip()

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

def write_header_once(filepath, header):
    """Write header if file does not exist or is empty"""
    if not os.path.exists(filepath) or os.stat(filepath).st_size == 0:
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

keypoint_names = list(upper_body_indices.keys())

# ===== CSV 1: Keypoint Positions =====
position_dir = os.path.join(data_dir, "positions")
os.makedirs(position_dir, exist_ok=True)
keypoints_csv = os.path.join(position_dir, f'keypoints_positions_{category}.csv')

keypoints_header = ['frame', 'timestamp', 'person_id']
for kp in keypoint_names:
    keypoints_header.extend([f'{kp}_x', f'{kp}_y', f'{kp}_conf'])
write_header_once(keypoints_csv, keypoints_header)

# ===== CSV 2: Distances Between Points =====
distance_dir = os.path.join(data_dir, "distances")
os.makedirs(distance_dir, exist_ok=True)
distances_csv = os.path.join(distance_dir, f'keypoints_distances_{category}.csv')

# Generate all possible pairs
keypoint_pairs = list(combinations(keypoint_names, 2))
distances_header = ['frame', 'timestamp', 'person_id']
for pair in keypoint_pairs:
    distances_header.append(f'dist_{pair[0]}_to_{pair[1]}')
write_header_once(distances_csv, distances_header)

print(f"Keypoints CSV: {keypoints_csv}")
print(f"Distances CSV: {distances_csv}")
print(f"Tracking {len(keypoint_names)} upper body points")
print(f"Computing {len(keypoint_pairs)} distance measurements")
print("\nPress 'q' to quit\n")

frame_count = 0

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = datetime.now().isoformat()

        # Run YOLOv8 pose detection
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        # Prepare CSV rows for this frame
        keypoints_rows = []
        distances_rows = []

        # Extract keypoints
        if results[0].keypoints is not None:
            all_keypoints = results[0].keypoints.data.cpu().numpy()

            for person_id, full_kps in enumerate(all_keypoints):
                upper_body_kps = {}
                keypoints_row = [frame_count, current_time, person_id]

                # Collect upper body keypoints
                for name, idx in upper_body_indices.items():
                    x, y, conf = full_kps[idx]
                    upper_body_kps[name] = (float(x), float(y), float(conf))
                    keypoints_row.extend([float(x), float(y), float(conf)])

                keypoints_rows.append(keypoints_row)

                # Calculate distances
                distances_row = [frame_count, current_time, person_id]
                for pair in keypoint_pairs:
                    p1 = upper_body_kps[pair[0]]
                    p2 = upper_body_kps[pair[1]]
                    if p1[2] > 0.5 and p2[2] > 0.5:
                        dist = calculate_distance(p1[:2], p2[:2])
                    else:
                        dist = -1.0
                    distances_row.append(float(dist))

                distances_rows.append(distances_row)

                # Label on frame
                if upper_body_kps['nose'][2] > 0.3:
                    cv2.putText(
                        annotated_frame,
                        f'Person {person_id}',
                        (int(upper_body_kps['nose'][0]), int(upper_body_kps['nose'][1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

        # Write all detected people data for this frame
        if keypoints_rows:
            with open(keypoints_csv, 'a', newline='') as f:
                csv.writer(f).writerows(keypoints_rows)
        if distances_rows:
            with open(distances_csv, 'a', newline='') as f:
                csv.writer(f).writerows(distances_rows)

        # Display frame info
        cv2.putText(annotated_frame, f'Frame: {frame_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Upper Body Points: {len(keypoint_names)}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated_frame, 'Press Q to quit', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('Upper Body Pose Detection', annotated_frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{'='*60}")
    print("Recording complete!")
    print(f"{'='*60}")
    print(f"Keypoints CSV: {keypoints_csv}")
    print(f"Distances CSV: {distances_csv}")
    print(f"Total frames processed: {frame_count}")
    print(f"Upper body keypoints tracked: {len(keypoint_names)}")
    print(f"Distance measurements per frame: {len(keypoint_pairs)}")
    print(f"{'='*60}")
