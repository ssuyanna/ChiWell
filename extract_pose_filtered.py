import cv2
import mediapipe as mp
import math
import json
import os
import numpy as np

# 保留6个角度
ANGLE_DEFS = [
    (11, 13, 15, "left_elbow_angle"),
    (12, 14, 16, "right_elbow_angle"),
    (23, 25, 27, "left_knee_angle"),
    (24, 26, 28, "right_knee_angle"),
    (11, 23, 25, "left_hip_angle"),
    (12, 24, 26, "right_hip_angle"),
]

ALLOWED_POINTS = set(range(11, 29))

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def process_video_filtered(video_path, output_json_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Cannot open: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_frames = []
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_number / fps
        frame_number += 1

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        frame_data = {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "landmarks": [],
            "angles": {}
        }

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            for i in ALLOWED_POINTS:
                pt = lm[i]
                frame_data["landmarks"].append({
                    "index": i,
                    "x": pt.x,
                    "y": pt.y,
                    "z": pt.z,
                    "visibility": pt.visibility
                })

            def get_xy(idx):
                pt = lm[idx]
                return [pt.x, pt.y]

            for a, b, c, name in ANGLE_DEFS:
                try:
                    if min(lm[a].visibility, lm[b].visibility, lm[c].visibility) > 0.5:
                        angle = calculate_angle(get_xy(a), get_xy(b), get_xy(c))
                        frame_data["angles"][name] = angle
                except:
                    frame_data["angles"][name] = None

        all_frames.append(frame_data)

    cap.release()
    pose.close()

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_frames, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {output_json_path}")
