import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from pathlib import Path

from .pose_extractor import PoseFeatureExtractor 

class VideoAnalyzer:

    def __init__(self):
        self.extractor = PoseFeatureExtractor()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        
        self.feature_names = [
            'left_arm_angle', 'right_arm_angle', 'left_knee_angle', 'right_knee_angle',
            'shoulders_inclination', 'hips_inclination', 'pelvis_angle'
        ]

    def generate_analysis_video(self, video_path, output_path, slowdown_factor=1.4):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Lỗi: Không thể mở file video {video_path}")
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        output_fps = fps / slowdown_factor
        
        out_width, out_height = 1280, 720
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (out_width, out_height))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for _ in tqdm(range(total_frames), desc=f"Analyzing {Path(video_path).name}"):
            ret, frame = cap.read()
            if not ret: break
            composite_frame = self._create_composite_frame(frame)
            out.write(composite_frame)

        cap.release()
        out.release()
        print(f"\nVideo kết quả được lưu tại: {output_path}")

    def _create_composite_frame(self, frame):
        panel_width, panel_height = 640, 360
        
        original_panel = cv2.resize(frame, (panel_width, panel_height))
        pose_panel = original_panel.copy()
        
        results = self.extractor.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        features = {}
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                pose_panel, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=2, circle_radius=3), 
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)     
            )
            features = self.extractor._calculate_geometric_features(results.pose_landmarks, frame.shape)

    
        table_panel = self._create_feature_table_panel(features)
        
       
        blank_panel = self._create_blank_panel(panel_width, panel_height)
        
        top_row = np.concatenate((original_panel, pose_panel), axis=1)
        bottom_row = np.concatenate((table_panel, blank_panel), axis=1) 
        return np.concatenate((top_row, bottom_row), axis=0)

    def _create_feature_table_panel(self, features):
        panel = np.ones((360, 640, 3), dtype=np.uint8) * 20
        y_pos = 50
        font_scale = 0.9
        
        for name in self.feature_names:
            text = f"{name.replace('_', ' ').title()}:"
            cv2.putText(panel, text, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
            
            if name in features and features[name] is not None:
                val_text = f"{features[name]:.1f}"
                cv2.putText(panel, val_text, (450, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
            
            y_pos += 45
        return panel
    
    def _create_blank_panel(self, width, height):
        return np.ones((height, width, 3), dtype=np.uint8) * 20
