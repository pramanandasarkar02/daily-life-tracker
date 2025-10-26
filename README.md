
# 🕜 Daily Life Tracker

A **personalized daily life activity tracker** that detects attention and inattention states using upper-body posture data captured through a webcam.  
This project is fully trained on **my own dataset**, making it unique and adaptive to your daily behavior patterns.

---

## ✨ Features

- Collects real-time pose keypoints using **YOLOv8n-pose**
- Calculates **upper-body keypoint distances**
- Trains a **Random Forest classifier** on your personal dataset
- Works as a **real-time application** built with **Tkinter**
- Designed for self-learning: the model is tuned only on **your own active/inactive frames**

---

## 🧩 Project Structure

```

daily-life-tracker/

-- data-processing/
   -- main.py              # Collects keypoints & distances using YOLOv8n-pose

-- model-train/
   -- main.py              # Trains RandomForest model on your dataset

-- inference/
   -- main.py              # Runs inference/testing on new frames

-- app/
   -- main.py              # Tkinter real-time attention detection app

-- dataset/
   -- distances/           # Stores collected CSV data
   -- positions/           # Stores Position of key points

-- models/
   -- attention_classifier.pkl  # Trained RandomForest model
   -- scaler.pkl                # StandardScaler for normalization

-- requirements.txt
-- LINCENSE.md
-- README.md

````

---

## ⚙️ Installation & Setup

```bash
# Create a virtual environment
python -m venv env

# Activate the environment
source env/bin/activate

# Install all dependencies
pip install -r requirements.txt
````

---

## 🚀 Usage Guide

### 1️⃣ Collect Dataset

Capture upper-body keypoints using your webcam.

```bash
python data-processing/main.py
```

* When prompted, enter the **category**:

  * `attention`
  * `inattention`

---

### 2️⃣ Train Model

Train a **Random Forest** classifier using your personal dataset.

```bash
python model-train/main.py
```

✅ This step saves:

* `./models/attention_classifier.pkl`
* `./models/scaler.pkl`

---

### 3️⃣ Run Inference (Testing)

Test the trained model on live or sample data.

```bash
python inference/main.py
```

---

### 4️⃣ Launch Real-Time App

Run the Tkinter GUI for real-time detection:

```bash
python app/main.py
```

---

## 🧠 Model Description

| Component               | Description                                      |
| ----------------------- | ------------------------------------------------ |
| **Pose Model**          | YOLOv8n-pose for keypoint extraction             |
| **Feature Engineering** | Euclidean distances between upper-body keypoints |
| **Classifier**          | RandomForestClassifier                           |
| **Scaler**              | StandardScaler (for consistent normalization)    |

---

## 📊 Dataset Summary

* **6000 frames (active/attention)**
* **6000 frames (inactive/inattention)**
* All data captured **from your webcam**
* Only **upper-body points** used for analysis

---

## 👤 Author

**Pramananda Sarkar(pramanandasarkar02)**
Personal project on *human attention modeling and daily activity tracking.*
- mail: **pramanandasarkar02@gmail.com**

---

## 📜 License

This project is developed for **personal use**.
You may reuse the structure and scripts for your own tracking applications. The Project Follow **MIT LICENSE**

