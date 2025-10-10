"""
Activity Detection System using YOLOv8 Pose + CNN Classifier
============================================================
This script provides three modes:
1. Dataset Collection - Collect pose keypoints for active/inactive positions
2. Model Training - Train a CNN classifier on your custom dataset
3. Real-time Inference - Detect activity percentage in real-time

Author: Activity Detector v1.0
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
import os
import json
from datetime import datetime
import pickle

# ============================================================================
# 1. POSE KEYPOINT EXTRACTOR
# ============================================================================

class PoseExtractor:
    """Extract and normalize pose keypoints from YOLO results"""
    
    def __init__(self, model_path="yolov8n-pose.pt"):
        self.model = YOLO(model_path)
        self.keypoint_dim = 17  # YOLOv8 pose has 17 keypoints
        
    def extract_keypoints(self, frame):
        """
        Extract normalized keypoints from frame
        Returns: numpy array of shape (51,) - 17 keypoints * 3 (x, y, confidence)
        """
        results = self.model(frame, imgsz=320, verbose=False)
        
        if len(results[0].keypoints) == 0:
            return None
        
        keypoints = results[0].keypoints.data[0].cpu().numpy()  # First person
        
        # Normalize keypoints relative to frame size
        h, w = frame.shape[:2]
        keypoints[:, 0] /= w  # Normalize x
        keypoints[:, 1] /= h  # Normalize y
        
        return keypoints.flatten()  # Shape: (51,)
    
    def get_annotated_frame(self, frame):
        """Get frame with pose annotations"""
        results = self.model(frame, imgsz=320, verbose=False)
        return results[0].plot()


# ============================================================================
# 2. DATASET COLLECTION
# ============================================================================

class DatasetCollector:
    """Collect and save pose dataset for active/inactive positions"""
    
    def __init__(self, save_dir="pose_dataset"):
        self.save_dir = save_dir
        self.active_dir = os.path.join(save_dir, "active")
        self.inactive_dir = os.path.join(save_dir, "inactive")
        
        os.makedirs(self.active_dir, exist_ok=True)
        os.makedirs(self.inactive_dir, exist_ok=True)
        
        self.extractor = PoseExtractor()
        
    def collect_data(self):
        """Interactive data collection session"""
        cap = cv2.VideoCapture(0)
        
        mode = "inactive"  # Start with inactive
        samples_collected = {"active": 0, "inactive": 0}
        
        print("\n" + "="*60)
        print("DATASET COLLECTION MODE")
        print("="*60)
        print("Instructions:")
        print("  - Press 'A' to switch to ACTIVE mode")
        print("  - Press 'I' to switch to INACTIVE mode")
        print("  - Press 'SPACE' to capture current pose")
        print("  - Press 'Q' to quit")
        print("="*60 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get annotated frame
            display_frame = self.extractor.get_annotated_frame(frame)
            
            # Add text overlay
            color = (0, 255, 0) if mode == "active" else (0, 0, 255)
            cv2.putText(display_frame, f"Mode: {mode.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(display_frame, f"Active: {samples_collected['active']} | Inactive: {samples_collected['inactive']}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "SPACE: Capture | A: Active | I: Inactive | Q: Quit", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            cv2.imshow("Dataset Collection", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('a'):
                mode = "active"
                print("Switched to ACTIVE mode")
            elif key == ord('i'):
                mode = "inactive"
                print("Switched to INACTIVE mode")
            elif key == ord(' '):
                # Capture pose
                keypoints = self.extractor.extract_keypoints(frame)
                if keypoints is not None:
                    save_path = os.path.join(
                        self.active_dir if mode == "active" else self.inactive_dir,
                        f"pose_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.npy"
                    )
                    np.save(save_path, keypoints)
                    samples_collected[mode] += 1
                    print(f"✓ Captured {mode} pose (Total: {samples_collected[mode]})")
                else:
                    print("✗ No person detected!")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("COLLECTION COMPLETE")
        print(f"Active samples: {samples_collected['active']}")
        print(f"Inactive samples: {samples_collected['inactive']}")
        print("="*60 + "\n")


# ============================================================================
# 3. CNN CLASSIFIER MODEL
# ============================================================================

class ActivityCNN(nn.Module):
    """CNN classifier for pose-based activity detection"""
    
    def __init__(self, input_dim=51):
        super(ActivityCNN, self).__init__()
        
        # Reshape input to use as 1D convolution
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)  # Binary classification: active/inactive
        )
        
    def forward(self, x):
        # x shape: (batch, 51)
        x = x.unsqueeze(1)  # (batch, 1, 51)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ============================================================================
# 4. DATASET LOADER
# ============================================================================

class PoseDataset(Dataset):
    """PyTorch dataset for pose keypoints"""
    
    def __init__(self, data_dir):
        self.samples = []
        self.labels = []
        
        # Load active samples
        active_dir = os.path.join(data_dir, "active")
        for file in os.listdir(active_dir):
            if file.endswith('.npy'):
                keypoints = np.load(os.path.join(active_dir, file))
                self.samples.append(keypoints)
                self.labels.append(1)  # Active = 1
        
        # Load inactive samples
        inactive_dir = os.path.join(data_dir, "inactive")
        for file in os.listdir(inactive_dir):
            if file.endswith('.npy'):
                keypoints = np.load(os.path.join(inactive_dir, file))
                self.samples.append(keypoints)
                self.labels.append(0)  # Inactive = 0
        
        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx]), torch.LongTensor([self.labels[idx]])[0]


# ============================================================================
# 5. MODEL TRAINER
# ============================================================================

class ModelTrainer:
    """Train the activity classification model"""
    
    def __init__(self, data_dir="pose_dataset", model_save_path="activity_model.pth"):
        self.data_dir = data_dir
        self.model_save_path = model_save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def train(self, epochs=50, batch_size=16, learning_rate=0.001):
        """Train the model"""
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        print(f"Device: {self.device}")
        
        # Load dataset
        dataset = PoseDataset(self.data_dir)
        print(f"Total samples: {len(dataset)}")
        
        if len(dataset) < 10:
            print("ERROR: Not enough samples! Collect at least 10 samples per class.")
            return
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        model = ActivityCNN().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        best_val_acc = 0.0
        
        print(f"Training samples: {train_size}, Validation samples: {val_size}")
        print("="*60 + "\n")
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss/len(train_loader):.4f} "
                  f"Train Acc: {train_acc:.2f}% "
                  f"Val Loss: {val_loss/len(val_loader):.4f} "
                  f"Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), self.model_save_path)
                print(f"✓ Model saved! (Best Val Acc: {best_val_acc:.2f}%)")
        
        print("\n" + "="*60)
        print(f"TRAINING COMPLETE - Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"Model saved to: {self.model_save_path}")
        print("="*60 + "\n")


# ============================================================================
# 6. REAL-TIME INFERENCE
# ============================================================================

class ActivityDetector:
    """Real-time activity detection"""
    
    def __init__(self, model_path="activity_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActivityCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.extractor = PoseExtractor()
        
        # Smoothing buffer
        self.buffer_size = 10
        self.prediction_buffer = []
        
    def predict(self, keypoints):
        """Predict activity from keypoints"""
        if keypoints is None:
            return None, 0.0
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(keypoints).unsqueeze(0).to(self.device)
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            return predicted.item(), confidence.item()
    
    def run_inference(self):
        """Run real-time inference"""
        cap = cv2.VideoCapture(0)
        
        print("\n" + "="*60)
        print("REAL-TIME ACTIVITY DETECTION")
        print("="*60)
        print("Press 'Q' to quit")
        print("="*60 + "\n")
        
        active_frames = 0
        total_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract keypoints
            keypoints = self.extractor.extract_keypoints(frame)
            
            # Get annotated frame
            display_frame = self.extractor.get_annotated_frame(frame)
            
            if keypoints is not None:
                # Predict
                prediction, confidence = self.predict(keypoints)
                
                # Update buffer for smoothing
                self.prediction_buffer.append(prediction)
                if len(self.prediction_buffer) > self.buffer_size:
                    self.prediction_buffer.pop(0)
                
                # Calculate smoothed prediction
                avg_prediction = np.mean(self.prediction_buffer)
                is_active = avg_prediction > 0.5
                
                # Update statistics
                total_frames += 1
                if is_active:
                    active_frames += 1
                
                activity_percentage = (active_frames / total_frames) * 100
                
                # Display information
                status = "ACTIVE" if is_active else "INACTIVE"
                color = (0, 255, 0) if is_active else (0, 0, 255)
                
                cv2.putText(display_frame, f"Status: {status}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                cv2.putText(display_frame, f"Confidence: {confidence*100:.1f}%", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Activity: {activity_percentage:.1f}%", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Activity bar
                bar_width = int((activity_percentage / 100) * 300)
                cv2.rectangle(display_frame, (10, 130), (310, 160), (50, 50, 50), -1)
                cv2.rectangle(display_frame, (10, 130), (10 + bar_width, 160), color, -1)
                cv2.putText(display_frame, f"{activity_percentage:.0f}%", (320, 155),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(display_frame, "No person detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Activity Detection", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("INFERENCE SESSION COMPLETE")
        print(f"Overall Activity: {activity_percentage:.1f}%")
        print("="*60 + "\n")


# ============================================================================
# 7. MAIN PROGRAM
# ============================================================================

def main():
    """Main program with menu system"""
    
    print("\n" + "="*60)
    print("ACTIVITY DETECTION SYSTEM")
    print("="*60)
    print("Select mode:")
    print("  1. Collect Dataset (Capture active/inactive poses)")
    print("  2. Train Model (Train CNN classifier)")
    print("  3. Run Real-time Detection")
    print("  4. Exit")
    print("="*60)
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        collector = DatasetCollector()
        collector.collect_data()
        
    elif choice == "2":
        epochs = int(input("Enter number of epochs (default 50): ") or "50")
        trainer = ModelTrainer()
        trainer.train(epochs=epochs)
        
    elif choice == "3":
        if not os.path.exists("activity_model.pth"):
            print("\nERROR: Model not found! Please train the model first (Option 2).")
            return
        detector = ActivityDetector()
        detector.run_inference()
        
    elif choice == "4":
        print("Exiting...")
        return
    
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()