import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# -------------------- Dataset --------------------
class VideoDataset(Dataset):
    def __init__(self, samples, labels, img_size=(128, 128), frames_per_clip=16, augment=False):
        self.img_size = img_size
        self.frames_per_clip = frames_per_clip
        self.samples = samples
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths = self.samples[idx]
        label = self.labels[idx]

        step = max(1, len(frame_paths) // self.frames_per_clip)
        selected = frame_paths[::step][:self.frames_per_clip]

        # Pad if not enough frames
        if len(selected) < self.frames_per_clip:
            selected = selected + [selected[-1]] * (self.frames_per_clip - len(selected))

        clip = []
        for f in selected:
            img = cv2.imread(f)
            if img is None:
                img = np.zeros((128, 128, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
            
            # Simple augmentation for training
            if self.augment and np.random.rand() > 0.5:
                img = cv2.flip(img, 1)  # Horizontal flip
            
            img = img / 255.0
            clip.append(img)

        clip = np.array(clip, dtype=np.float32)  # (T, H, W, C)
        clip = np.transpose(clip, (3, 0, 1, 2))  # (C, T, H, W)
        return torch.tensor(clip, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def load_dataset(data_dir, img_size=(128, 128), frames_per_clip=16):
    """Load and split dataset into train/val"""
    samples = []
    labels = []
    label_encoder = LabelEncoder()

    all_labels = []
    for label in sorted(os.listdir(data_dir)):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        frames = sorted(os.listdir(label_path))
        if len(frames) < frames_per_clip:
            print(f"Warning: {label} has only {len(frames)} frames, needs at least {frames_per_clip}")
            continue
        samples.append([os.path.join(label_path, f) for f in frames])
        all_labels.append(label)

    if len(samples) == 0:
        raise ValueError("No valid samples found!")

    labels = label_encoder.fit_transform(all_labels)
    
    print(f"Total samples: {len(samples)}")
    print(f"Classes: {label_encoder.classes_}")
    print(f"Samples per class: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # Check if we have enough samples to split
    unique, counts = np.unique(labels, return_counts=True)
    min_samples = counts.min()
    
    if min_samples < 2:
        print(f"\nWARNING: Some classes have only {min_samples} sample(s). Need at least 2 per class.")
        print("Using all data for training (no validation split).")
        train_samples, train_labels = samples, labels
        val_samples, val_labels = [], []
    elif len(samples) < 10:
        print(f"\nWARNING: Only {len(samples)} total samples. Using all for training (no validation).")
        train_samples, train_labels = samples, labels
        val_samples, val_labels = [], []
    else:
        # Split into train/val
        train_samples, val_samples, train_labels, val_labels = train_test_split(
            samples, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    train_dataset = VideoDataset(train_samples, train_labels, img_size, frames_per_clip, augment=True)
    val_dataset = VideoDataset(val_samples, val_labels, img_size, frames_per_clip, augment=False) if val_samples else None
    
    return train_dataset, val_dataset, label_encoder


# -------------------- Model --------------------
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

        # Fix: Use batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True  # This fixes the warning
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x):
        feats = self.cnn3d(x)           # (B, E, T, 1, 1)
        feats = feats.squeeze(-1).squeeze(-1)  # (B, E, T)
        feats = feats.permute(0, 2, 1)  # (B, T, E) - batch_first format
        out = self.transformer(feats)   # (B, T, E)
        out = out.mean(1)               # (B, E) - average over time
        return self.fc(out)


# -------------------- Train --------------------
def train_model(data_dir, epochs=20, batch_size=4, lr=1e-3):
    train_dataset, val_dataset, label_encoder = load_dataset(data_dir)
    num_classes = len(label_encoder.classes_)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2) if val_dataset else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ActionRecognitionModel(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(clips)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = outputs.max(1)
            train_correct += pred.eq(labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}")
        
        # Validation (if available)
        if val_loader:
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for clips, labels in val_loader:
                    clips, labels = clips.to(device), labels.to(device)
                    outputs = model(clips)
                    loss = F.cross_entropy(outputs, labels)
                    
                    val_loss += loss.item()
                    _, pred = outputs.max(1)
                    val_correct += pred.eq(labels).sum().item()
                    val_total += labels.size(0)

            val_acc = val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            scheduler.step(avg_val_loss)
            
            print(f"  Val   - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'num_classes': num_classes,
                    'classes': label_encoder.classes_.tolist()
                }, "best_activity_model.pth")
                # Save label encoder separately
                import pickle
                with open("label_encoder.pkl", "wb") as f:
                    pickle.dump(label_encoder, f)
                print(f"  → Best model saved (val_acc: {val_acc:.4f})")
        else:
            # No validation - just save periodically
            scheduler.step(avg_train_loss)
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'train_acc': train_acc,
                    'num_classes': num_classes,
                    'classes': label_encoder.classes_.tolist()
                }, "best_activity_model.pth")
                # Save label encoder separately
                import pickle
                with open("label_encoder.pkl", "wb") as f:
                    pickle.dump(label_encoder, f)
                print(f"  → Model saved")

    return model, label_encoder


# -------------------- Inference --------------------
def predict(model, encoder, video_dir, frames_per_clip=16, img_size=(128, 128)):
    model.eval()
    device = next(model.parameters()).device
    
    frames = sorted([f for f in os.listdir(video_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    step = max(1, len(frames) // frames_per_clip)
    selected = frames[::step][:frames_per_clip]
    
    # Pad if needed
    if len(selected) < frames_per_clip:
        selected = selected + [selected[-1]] * (frames_per_clip - len(selected))

    clip = []
    for f in selected:
        img = cv2.imread(os.path.join(video_dir, f))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img / 255.0
        clip.append(img)

    clip = np.array(clip, dtype=np.float32)
    clip = np.transpose(clip, (3, 0, 1, 2))
    clip = torch.tensor(clip, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        outputs = model(clip.to(device))
        probs = F.softmax(outputs, dim=1)
        pred = outputs.argmax(1).item()
        confidence = probs[0, pred].item()
    
    return encoder.inverse_transform([pred])[0], confidence


# -------------------- Main --------------------
def main():
    data_dir = "data/dataset"  # Your dataset path
    
    # Train model
    model, encoder = train_model(data_dir, epochs=20, batch_size=4, lr=1e-3)

    # Load best model for inference
    checkpoint = torch.load("best_activity_model.pth", weights_only=True)
    
    # Recreate label encoder from saved classes
    import pickle
    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    
    # Recreate model with correct number of classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionRecognitionModel(num_classes=checkpoint['num_classes']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    metric_key = 'val_acc' if 'val_acc' in checkpoint else 'train_acc'
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1} with {metric_key}: {checkpoint[metric_key]:.4f}")

    # Inference example
    test_video_dir = os.path.join(data_dir, sorted(os.listdir(data_dir))[0])
    if os.path.isdir(test_video_dir):
        prediction, confidence = predict(model, encoder, test_video_dir)
        print(f"\nPredicted activity: {prediction} (confidence: {confidence:.4f})")


if __name__ == "__main__":
    main()