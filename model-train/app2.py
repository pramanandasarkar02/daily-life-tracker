import cv2
import torch
import numpy as np
import pickle
from collections import deque
import torch.nn as nn
import torch.nn.functional as F


# -------------------- Model (same as training) --------------------
class ActionRecognitionModel(nn.Module):
    def __init__(self, num_classes, embed_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()

        self.cnn3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Dropout3d(0.1),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Dropout3d(0.1),

            nn.Conv3d(128, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x):
        feats = self.cnn3d(x)
        feats = feats.squeeze(-1).squeeze(-1)
        feats = feats.permute(0, 2, 1)
        out = self.transformer(feats)
        out = out.mean(1)
        return self.fc(out)


# -------------------- Activity Status Mapper --------------------
def map_to_activity_status(predicted_class, confidence, motion_score):
    """
    Map predicted class to activity status: ACTIVE, INACTIVE, or OFFLINE
    
    Args:
        predicted_class: The class predicted by the model
        confidence: Prediction confidence (0-1)
        motion_score: Amount of motion detected in frames (0-1)
    """
    # You can customize these mappings based on your trained classes
    # Example: if you have classes like 'walking', 'running', 'sitting', 'standing', etc.
    
    active_keywords = ['walking', 'running', 'jumping', 'exercising', 'playing', 'dancing']
    inactive_keywords = ['sitting', 'standing', 'lying', 'sleeping', 'reading']
    
    predicted_lower = predicted_class.lower()
    
    # Check if person is detected with reasonable confidence
    if confidence < 0.3 or motion_score < 0.05:
        return "OFFLINE", (128, 128, 128)  # Gray
    
    # Check for active activities
    for keyword in active_keywords:
        if keyword in predicted_lower:
            return "ACTIVE", (0, 255, 0)  # Green
    
    # Check for inactive activities
    for keyword in inactive_keywords:
        if keyword in predicted_lower:
            return "INACTIVE", (255, 165, 0)  # Orange
    
    # Default based on motion score
    if motion_score > 0.3:
        return "ACTIVE", (0, 255, 0)
    elif motion_score > 0.1:
        return "INACTIVE", (255, 165, 0)
    else:
        return "OFFLINE", (128, 128, 128)


# -------------------- Real-time Inference --------------------
class WebcamActivityMonitor:
    def __init__(self, model_path, label_encoder_path, frames_per_clip=16, img_size=(128, 128)):
        self.frames_per_clip = frames_per_clip
        self.img_size = img_size
        self.frame_buffer = deque(maxlen=frames_per_clip)
        self.prev_frame = None
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # Load label encoder
        with open(label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)
        
        # Initialize model
        self.model = ActionRecognitionModel(num_classes=checkpoint['num_classes']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Classes: {self.label_encoder.classes_}")
    
    def calculate_motion_score(self, frame):
        """Calculate motion between consecutive frames"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return 0.0
        
        # Calculate frame difference
        diff = cv2.absdiff(self.prev_frame, gray)
        motion_score = np.mean(diff) / 255.0
        
        self.prev_frame = gray
        return motion_score
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, self.img_size)
        frame_normalized = frame_resized / 255.0
        return frame_normalized
    
    def predict(self):
        """Make prediction on buffered frames"""
        if len(self.frame_buffer) < self.frames_per_clip:
            return None, 0.0
        
        # Convert buffer to tensor
        clip = np.array(list(self.frame_buffer), dtype=np.float32)
        clip = np.transpose(clip, (3, 0, 1, 2))  # (C, T, H, W)
        clip_tensor = torch.tensor(clip, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(clip_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_idx = outputs.argmax(1).item()
            confidence = probs[0, pred_idx].item()
        
        predicted_class = self.label_encoder.inverse_transform([pred_idx])[0]
        return predicted_class, confidence
    
    def draw_status(self, frame, status, color, predicted_class, confidence, motion_score, fps):
        """Draw status overlay on frame"""
        h, w = frame.shape[:2]
        
        # Draw semi-transparent overlay at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
        # Draw status with large text (use thickness for bold effect)
        cv2.putText(frame, f"STATUS: {status}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Draw additional info
        cv2.putText(frame, f"Activity: {predicted_class}", (20, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2%}", (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw motion indicator
        motion_bar_width = int(200 * motion_score)
        cv2.rectangle(frame, (w - 220, 20), (w - 20, 40), (50, 50, 50), -1)
        cv2.rectangle(frame, (w - 220, 20), (w - 220 + motion_bar_width, 40), (0, 255, 255), -1)
        cv2.putText(frame, "Motion", (w - 220, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    def run(self, camera_index=0, prediction_interval=8):
        """
        Run real-time monitoring
        
        Args:
            camera_index: Webcam index (0 for default)
            prediction_interval: Number of frames between predictions (lower = more frequent)
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Starting webcam monitoring...")
        print("Press 'q' to quit")
        
        frame_count = 0
        current_prediction = "Initializing..."
        current_confidence = 0.0
        current_status = "OFFLINE"
        current_color = (128, 128, 128)
        motion_score = 0.0
        
        # For FPS calculation
        fps_time = cv2.getTickCount()
        fps = 0.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                current_status = "OFFLINE"
                current_color = (128, 128, 128)
                print("Warning: Failed to grab frame")
                break
            
            # Calculate motion score
            motion_score = self.calculate_motion_score(frame)
            
            # Add frame to buffer
            processed_frame = self.preprocess_frame(frame)
            self.frame_buffer.append(processed_frame)
            
            # Make prediction at intervals
            if frame_count % prediction_interval == 0 and len(self.frame_buffer) == self.frames_per_clip:
                predicted_class, confidence = self.predict()
                if predicted_class:
                    current_prediction = predicted_class
                    current_confidence = confidence
                    current_status, current_color = map_to_activity_status(
                        predicted_class, confidence, motion_score
                    )
            
            # Calculate FPS
            fps_time_new = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (fps_time_new - fps_time)
            fps_time = fps_time_new
            
            # Draw status overlay
            display_frame = self.draw_status(
                frame, current_status, current_color, 
                current_prediction, current_confidence, motion_score, fps
            )
            
            # Display
            cv2.imshow('Activity Monitor', display_frame)
            
            frame_count += 1
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Monitoring stopped")


# -------------------- Main --------------------
def main():
    model_path = "best_activity_model.pth"
    label_encoder_path = "label_encoder.pkl"
    
    try:
        monitor = WebcamActivityMonitor(
            model_path=model_path,
            label_encoder_path=label_encoder_path,
            frames_per_clip=16,
            img_size=(128, 128)
        )
        
        # Start monitoring
        # You can change camera_index if you have multiple cameras
        # prediction_interval: lower = more frequent predictions but higher CPU usage
        monitor.run(camera_index=0, prediction_interval=8)
        
    except FileNotFoundError:
        print(f"Error: Model files not found!")
        print(f"Please ensure '{model_path}' and '{label_encoder_path}' exist.")
        print("Train the model first using the training script.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()