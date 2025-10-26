import cv2
from ultralytics import YOLO
import numpy as np
from itertools import combinations
import joblib
from collections import deque
import time
import json
import os
from datetime import datetime
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd

# ==================== DATA LOGGING CLASS ====================
class AttentionLogger:
    def __init__(self, log_file="attention_log.json"):
        self.log_file = log_file
        self.current_status = "offline"
        self.session_start = None
        self.sessions = []
        self.load_sessions()
    
    def load_sessions(self):
        """Load existing sessions from file"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    self.sessions = data.get('sessions', [])
            except:
                self.sessions = []
    
    def save_sessions(self):
        """Save sessions to file"""
        data = {
            'sessions': self.sessions,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.log_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def update_status(self, new_status):
        """Update current status and log session"""
        now = datetime.now()
        
        # If status changed, log the previous session
        if new_status != self.current_status:
            if self.session_start is not None:
                duration = (now - self.session_start).total_seconds()
                session = {
                    'status': self.current_status,
                    'start_time': self.session_start.isoformat(),
                    'end_time': now.isoformat(),
                    'duration': duration
                }
                self.sessions.append(session)
                self.save_sessions()
            
            self.current_status = new_status
            self.session_start = now
    
    def get_statistics(self):
        """Calculate statistics from sessions"""
        stats = {
            'attention': 0,
            'inattention': 0,
            'offline': 0,
            'total': 0
        }
        
        for session in self.sessions:
            duration = session['duration']
            status = session['status']
            stats[status] = stats.get(status, 0) + duration
            stats['total'] += duration
        
        return stats

# ==================== GUI DASHBOARD CLASS ====================
class DashboardApp:
    def __init__(self, root, logger):
        self.root = root
        self.logger = logger
        self.root.title("Attention Tracker Dashboard")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Create main container
        main_frame = tk.Frame(root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title = tk.Label(main_frame, text="Real-Time Attention Tracker", 
                        font=('Arial', 24, 'bold'), bg='#2c3e50', fg='white')
        title.pack(pady=10)
        
        # Status Frame
        self.create_status_frame(main_frame)
        
        # Stats Frame
        self.create_stats_frame(main_frame)
        
        # Chart Frame
        self.create_chart_frame(main_frame)
        
        # Sessions Frame
        self.create_sessions_frame(main_frame)
        
        # Control Buttons
        self.create_control_buttons(main_frame)
        
        # Start update loop
        self.update_dashboard()
    
    def create_status_frame(self, parent):
        """Create current status display"""
        status_frame = tk.LabelFrame(parent, text="Current Status", 
                                     font=('Arial', 12, 'bold'),
                                     bg='#34495e', fg='white', padx=10, pady=10)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = tk.Label(status_frame, text="OFFLINE", 
                                     font=('Arial', 32, 'bold'),
                                     bg='#34495e', fg='#95a5a6')
        self.status_label.pack()
        
        self.time_label = tk.Label(status_frame, text="Duration: 0s", 
                                   font=('Arial', 14),
                                   bg='#34495e', fg='white')
        self.time_label.pack()
    
    def create_stats_frame(self, parent):
        """Create statistics display"""
        stats_frame = tk.Frame(parent, bg='#2c3e50')
        stats_frame.pack(fill=tk.X, pady=5)
        
        # Attention
        att_frame = tk.LabelFrame(stats_frame, text="Attention", 
                                 font=('Arial', 10, 'bold'),
                                 bg='#27ae60', fg='white', padx=10, pady=10)
        att_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.att_time = tk.Label(att_frame, text="0h 0m 0s", 
                                font=('Arial', 16, 'bold'),
                                bg='#27ae60', fg='white')
        self.att_time.pack()
        
        self.att_percent = tk.Label(att_frame, text="0%", 
                                   font=('Arial', 12),
                                   bg='#27ae60', fg='white')
        self.att_percent.pack()
        
        # Inattention
        inatt_frame = tk.LabelFrame(stats_frame, text="Inattention", 
                                   font=('Arial', 10, 'bold'),
                                   bg='#e74c3c', fg='white', padx=10, pady=10)
        inatt_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.inatt_time = tk.Label(inatt_frame, text="0h 0m 0s", 
                                  font=('Arial', 16, 'bold'),
                                  bg='#e74c3c', fg='white')
        self.inatt_time.pack()
        
        self.inatt_percent = tk.Label(inatt_frame, text="0%", 
                                     font=('Arial', 12),
                                     bg='#e74c3c', fg='white')
        self.inatt_percent.pack()
        
        # Offline
        off_frame = tk.LabelFrame(stats_frame, text="Offline", 
                                 font=('Arial', 10, 'bold'),
                                 bg='#95a5a6', fg='white', padx=10, pady=10)
        off_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.off_time = tk.Label(off_frame, text="0h 0m 0s", 
                                font=('Arial', 16, 'bold'),
                                bg='#95a5a6', fg='white')
        self.off_time.pack()
        
        self.off_percent = tk.Label(off_frame, text="0%", 
                                   font=('Arial', 12),
                                   bg='#95a5a6', fg='white')
        self.off_percent.pack()
    
    def create_chart_frame(self, parent):
        """Create chart display"""
        chart_frame = tk.LabelFrame(parent, text="Activity Timeline", 
                                   font=('Arial', 12, 'bold'),
                                   bg='#34495e', fg='white')
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.figure = Figure(figsize=(10, 4), facecolor='#34495e')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#2c3e50')
        
        self.canvas = FigureCanvasTkAgg(self.figure, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_sessions_frame(self, parent):
        """Create sessions list"""
        sessions_frame = tk.LabelFrame(parent, text="Recent Sessions", 
                                      font=('Arial', 12, 'bold'),
                                      bg='#34495e', fg='white')
        sessions_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create Treeview
        columns = ('Status', 'Start Time', 'Duration')
        self.tree = ttk.Treeview(sessions_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150)
        
        scrollbar = ttk.Scrollbar(sessions_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_control_buttons(self, parent):
        """Create control buttons"""
        btn_frame = tk.Frame(parent, bg='#2c3e50')
        btn_frame.pack(pady=10)
        
        export_btn = tk.Button(btn_frame, text="Export Data", 
                              command=self.export_data,
                              bg='#3498db', fg='white', 
                              font=('Arial', 12, 'bold'),
                              padx=20, pady=10)
        export_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(btn_frame, text="Clear Data", 
                             command=self.clear_data,
                             bg='#e74c3c', fg='white', 
                             font=('Arial', 12, 'bold'),
                             padx=20, pady=10)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        refresh_btn = tk.Button(btn_frame, text="Refresh", 
                               command=self.refresh_dashboard,
                               bg='#2ecc71', fg='white', 
                               font=('Arial', 12, 'bold'),
                               padx=20, pady=10)
        refresh_btn.pack(side=tk.LEFT, padx=5)
    
    def format_duration(self, seconds):
        """Format duration in seconds to readable string"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"
    
    def update_dashboard(self):
        """Update dashboard with latest data"""
        # Update current status
        status = self.logger.current_status.upper()
        
        if status == "ATTENTION":
            color = '#27ae60'
        elif status == "INATTENTION":
            color = '#e74c3c'
        else:
            color = '#95a5a6'
        
        self.status_label.config(text=status, fg=color)
        
        # Update duration
        if self.logger.session_start:
            duration = (datetime.now() - self.logger.session_start).total_seconds()
            self.time_label.config(text=f"Duration: {self.format_duration(duration)}")
        
        # Update statistics
        stats = self.logger.get_statistics()
        total = stats['total'] if stats['total'] > 0 else 1
        
        self.att_time.config(text=self.format_duration(stats['attention']))
        self.att_percent.config(text=f"{(stats['attention']/total*100):.1f}%")
        
        self.inatt_time.config(text=self.format_duration(stats['inattention']))
        self.inatt_percent.config(text=f"{(stats['inattention']/total*100):.1f}%")
        
        self.off_time.config(text=self.format_duration(stats['offline']))
        self.off_percent.config(text=f"{(stats['offline']/total*100):.1f}%")
        
        # Update chart
        self.update_chart()
        
        # Update sessions list
        self.update_sessions_list()
        
        # Schedule next update
        self.root.after(1000, self.update_dashboard)
    
    def update_chart(self):
        """Update the timeline chart"""
        self.ax.clear()
        
        if len(self.logger.sessions) == 0:
            self.ax.text(0.5, 0.5, 'No data yet', 
                        ha='center', va='center',
                        color='white', fontsize=14)
            self.canvas.draw()
            return
        
        # Prepare data
        df = pd.DataFrame(self.logger.sessions)
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['hour'] = df['start_time'].dt.hour
        
        # Group by hour and status
        hourly = df.groupby(['hour', 'status'])['duration'].sum().unstack(fill_value=0)
        
        # Plot
        if 'attention' in hourly.columns:
            self.ax.plot(hourly.index, hourly['attention'], 
                        label='Attention', color='#27ae60', linewidth=2, marker='o')
        if 'inattention' in hourly.columns:
            self.ax.plot(hourly.index, hourly['inattention'], 
                        label='Inattention', color='#e74c3c', linewidth=2, marker='o')
        if 'offline' in hourly.columns:
            self.ax.plot(hourly.index, hourly['offline'], 
                        label='Offline', color='#95a5a6', linewidth=2, marker='o')
        
        self.ax.set_xlabel('Hour of Day', color='white', fontsize=10)
        self.ax.set_ylabel('Duration (seconds)', color='white', fontsize=10)
        self.ax.tick_params(colors='white')
        self.ax.legend(facecolor='#34495e', edgecolor='white', labelcolor='white')
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def update_sessions_list(self):
        """Update the sessions list"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add recent sessions (last 20)
        recent = self.logger.sessions[-20:]
        for session in reversed(recent):
            status = session['status'].capitalize()
            start = datetime.fromisoformat(session['start_time']).strftime('%H:%M:%S')
            duration = self.format_duration(session['duration'])
            
            self.tree.insert('', 0, values=(status, start, duration))
    
    def export_data(self):
        """Export data to CSV"""
        if len(self.logger.sessions) == 0:
            messagebox.showinfo("Export", "No data to export")
            return
        
        filename = f"attention_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(self.logger.sessions)
        df.to_csv(filename, index=False)
        messagebox.showinfo("Export", f"Data exported to {filename}")
    
    def clear_data(self):
        """Clear all data"""
        if messagebox.askyesno("Clear Data", "Are you sure you want to clear all data?"):
            self.logger.sessions = []
            self.logger.save_sessions()
            self.refresh_dashboard()
            messagebox.showinfo("Clear", "Data cleared successfully")
    
    def refresh_dashboard(self):
        """Refresh the dashboard"""
        self.logger.load_sessions()
        self.update_dashboard()

# ==================== MAIN DETECTION SYSTEM ====================
class AttentionDetectionSystem:
    def __init__(self, logger):
        self.logger = logger
        
        # Load models
        print("Loading YOLO model...")
        self.model = YOLO('models/yolov8n-pose.pt')
        
        print("Loading attention classifier...")
        self.classifier = joblib.load("./models/attention_classifier.pkl")
        self.scaler = joblib.load("./models/scaler.pkl")
        
        # Upper body keypoint indices
        self.upper_body_indices = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2,
            'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10
        }
        
        self.keypoint_names = list(self.upper_body_indices.keys())
        self.keypoint_pairs = list(combinations(self.keypoint_names, 2))
        
        # Smoothing buffer
        self.prediction_buffer = deque(maxlen=10)
        
        # Frame counter
        self.frame_count = 0
        self.fps = 0
        self.fps_time = time.time()
        
        # Running flag
        self.running = True
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def extract_features(self, keypoints_data):
        """Extract distance features"""
        upper_body_kps = {}
        
        for name, idx in self.upper_body_indices.items():
            x, y, conf = keypoints_data[idx]
            upper_body_kps[name] = (float(x), float(y), float(conf))
        
        features = []
        for pair in self.keypoint_pairs:
            p1 = upper_body_kps[pair[0]]
            p2 = upper_body_kps[pair[1]]
            
            if p1[2] > 0.5 and p2[2] > 0.5:
                dist = self.calculate_distance(p1[:2], p2[:2])
            else:
                dist = 0
            
            features.append(float(dist))
        
        return features
    
    def get_smoothed_prediction(self, current_pred):
        """Smooth predictions"""
        self.prediction_buffer.append(current_pred)
        return np.mean(self.prediction_buffer) > 0.5
    
    def run(self):
        """Run detection system"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            return
        
        print("\n" + "="*60)
        print("Attention Detection System Running")
        print("Press 'q' to quit | 'r' to reset buffer")
        print("="*60 + "\n")
        
        # Start with offline status
        self.logger.update_status("offline")
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Calculate FPS
                if self.frame_count % 10 == 0:
                    self.fps = 10 / (time.time() - self.fps_time)
                    self.fps_time = time.time()
                
                # Run detection
                results = self.model(frame, verbose=False)
                annotated_frame = results[0].plot()
                
                person_detected = False
                
                # Process keypoints
                if results[0].keypoints is not None:
                    all_keypoints = results[0].keypoints.data.cpu().numpy()
                    
                    if len(all_keypoints) > 0:
                        person_detected = True
                        
                        # Process first person only
                        full_kps = all_keypoints[0]
                        
                        try:
                            features = self.extract_features(full_kps)
                            features_array = np.array(features).reshape(1, -1)
                            features_scaled = self.scaler.transform(features_array)
                            
                            prediction = self.classifier.predict(features_scaled)[0]
                            probability = self.classifier.predict_proba(features_scaled)[0]
                            
                            smoothed_pred = self.get_smoothed_prediction(prediction)
                            
                            if smoothed_pred:
                                label = "ATTENTION"
                                color = (0, 255, 0)
                                conf = probability[1]
                                self.logger.update_status("attention")
                            else:
                                label = "INATTENTION"
                                color = (0, 0, 255)
                                conf = probability[0]
                                self.logger.update_status("inattention")
                            
                            # Draw label
                            nose_x, nose_y, nose_conf = full_kps[0]
                            
                            if nose_conf > 0.3:
                                text = f"{label} ({conf:.2f})"
                                cv2.putText(annotated_frame, text,
                                          (int(nose_x)-80, int(nose_y)-60),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        
                        except Exception as e:
                            print(f"Error processing: {e}")
                
                # Update status if no person detected
                if not person_detected:
                    self.logger.update_status("offline")
                    cv2.putText(annotated_frame, "NO PERSON DETECTED",
                              (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Display info
                cv2.putText(annotated_frame, f'FPS: {self.fps:.1f}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f'Status: {self.logger.current_status.upper()}', 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Attention Detection', annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.prediction_buffer.clear()
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\nSystem stopped")
            print(f"Total frames: {self.frame_count}")

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Initialize logger
    logger = AttentionLogger("attention_log.json")
    
    # Create detection system
    detection_system = AttentionDetectionSystem(logger)
    
    # Start detection in separate thread
    detection_thread = threading.Thread(target=detection_system.run, daemon=True)
    detection_thread.start()
    
    # Start GUI dashboard
    root = tk.Tk()
    app = DashboardApp(root, logger)
    root.mainloop()
    
    # Stop detection when GUI closes
    detection_system.running = False
    detection_thread.join(timeout=2)