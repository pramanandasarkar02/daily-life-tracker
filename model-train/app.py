import os
import cv2
import torch
import numpy as np
import pickle
import torch.nn.functional as F

# -------------------- Model --------------------
class ActionRecognitionModel(torch.nn.Module):
    def __init__(self, num_classes, embed_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()

        self.cnn3d = torch.nn.Sequential(
            torch.nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d((1, 2, 2)),
            torch.nn.Dropout3d(0.1),

            torch.nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d((2, 2, 2)),
            torch.nn.Dropout3d(0.1),

            torch.nn.Conv3d(128, embed_dim, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm3d(embed_dim),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool3d((None, 1, 1))
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x):
        feats = self.cnn3d(x)           # (B, E, T, 1, 1)
        feats = feats.squeeze(-1).squeeze(-1)  # (B, E, T)
        feats = feats.permute(0, 2, 1)  # (B, T, E) - batch_first format
        out = self.transformer(feats)   # (B, T, E)
        out = out.mean(1)               # (B, E) - average over time
        return self.fc(out)

# -------------------- Load Model and Encoder --------------------
def load_model_and_encoder(model_path="best_activity_model.pth", encoder_path="label_encoder.pkl"):
    # Load label encoder
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, weights_only=True)
    num_classes = checkpoint['num_classes']
    
    # Recreate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionRecognitionModel(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    metric_key = 'val_acc' if 'val_acc' in checkpoint else 'train_acc'
    print(f"Loaded model from epoch {checkpoint['epoch']+1} with {metric_key}: {checkpoint[metric_key]:.4f}")
    
    return model, encoder, device

# -------------------- Real-time Prediction --------------------
def predict_on_clip(model, encoder, device, clip_frames, frames_per_clip=16, img_size=(128, 128)):
    # Select frames
    step = max(1, len(clip_frames) // frames_per_clip)
    selected = clip_frames[::step][:frames_per_clip]
    
    # Pad if needed
    if len(selected) < frames_per_clip:
        selected += [selected[-1]] * (frames_per_clip - len(selected))
    
    clip = []
    for img in selected:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img / 255.0
        clip.append(img)
    
    clip = np.array(clip, dtype=np.float32)
    clip = np.transpose(clip, (3, 0, 1, 2))
    clip_tensor = torch.tensor(clip).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(clip_tensor)
        probs = F.softmax(outputs, dim=1)
        pred = outputs.argmax(1).item()
        confidence = probs[0, pred].item()
    
    label = encoder.inverse_transform([pred])[0]
    return label, confidence

# -------------------- Main: Webcam Tracking --------------------
def main():
    model, encoder, device = load_model_and_encoder()
    
    # Assuming classes include 'ACTIVE', 'INACTIVE', 'OFFLINE'
    expected_classes = ['ACTIVE', 'INACTIVE', 'OFFLINE']
    actual_classes = encoder.classes_.tolist()
    if not all(cls in actual_classes for cls in expected_classes):
        print(f"Warning: Model classes {actual_classes} do not match expected {expected_classes}. Predictions may not align.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    frames_per_clip = 16
    buffer = []  # Buffer to hold recent frames
    frame_count = 0
    predict_interval = 15  # Predict every 15 frames (~0.5s at 30fps)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        buffer.append(frame)
        if len(buffer) > frames_per_clip * 2:  # Keep buffer reasonable size
            buffer = buffer[-frames_per_clip * 2:]
        
        frame_count += 1
        
        label = "Gathering frames..."
        confidence = 0.0
        
        if len(buffer) >= frames_per_clip and frame_count % predict_interval == 0:
            label, confidence = predict_on_clip(model, encoder, device, buffer, frames_per_clip)
        
        # Display prediction on frame
        display_text = f"{label} ({confidence:.2f})"
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Activity Tracker', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()