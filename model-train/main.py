import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs("./models", exist_ok=True)

# Load data
att = pd.read_csv("./dataset/distances/keypoints_distances_attention.csv")
inatt = pd.read_csv("./dataset/distances/keypoints_distances_inattention.csv")

# Add labels
att["label"] = 1   # attention
inatt["label"] = 0 # inattention

# Combine
data = pd.concat([att, inatt], ignore_index=True)

# Drop non-feature columns
data = data.drop(columns=["frame", "timestamp", "person_id"], errors="ignore")

# Split into X, y
X = data.drop(columns=["label"])
y = data["label"]

# Handle NaNs (replace with 0)
X = X.fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Inattention", "Attention"]))

# Save model and scaler in ./models directory
joblib.dump(model, "./models/attention_classifier.pkl")
joblib.dump(scaler, "./models/scaler.pkl")

print("Model saved as ./models/attention_classifier.pkl")
print("Scaler saved as ./models/scaler.pkl")