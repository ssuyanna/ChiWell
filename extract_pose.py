import cv2
import mediapipe as mp
import math
import json
import time
import os

# --- Configuration ---
VIDEO_PATH = "/Users/suya/Desktop/Baduanjin/baduanjin.mp4"  # <--- 修改为你的标准视频路径
OUTPUT_JSON_PATH = "standard_pose_data.json"
# 选择要计算的角度 (可以根据八段锦动作特点添加或修改)
# 每个元组代表一个角度: (关节点1, 顶点关节点, 关节点2, 角度名称)
# 关节点索引参考 MediaPipe Pose 文档: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#pose_landmarks
ANGLES_TO_CALCULATE = [
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.LEFT_WRIST, "left_elbow_angle"),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_WRIST, "right_elbow_angle"),
    (mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP, "left_shoulder_angle"),
    (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP, "right_shoulder_angle"),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_KNEE, "left_hip_angle"),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_KNEE, "right_hip_angle"),
    (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_KNEE, mp.solutions.pose.PoseLandmark.LEFT_ANKLE, "left_knee_angle"),
    (mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_KNEE, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE, "right_knee_angle"),
]

# --- Helper Function ---
def calculate_angle(landmark1, landmark2, landmark3):
    """计算由三个点构成的角度 (返回度数)"""
    # 获取坐标
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y
    x3, y3 = landmark3.x, landmark3.y

    # 计算角度的弧度值
    # 使用 atan2 来获得更精确的角度，并处理垂直线的情况
    angle_rad = math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)

    # 将弧度转换为度数
    angle_deg = math.degrees(angle_rad)

    # 使角度保持在 0 到 360 度之间 (或者根据需要调整为 -180 到 180)
    # angle_deg = abs(angle_deg) # 取绝对值可能更常用
    if angle_deg < 0:
        angle_deg += 360
    # 或者，如果你想要内角（通常小于180度）
    if angle_deg > 180:
         angle_deg = 360 - angle_deg

    return angle_deg

# --- Main Processing Logic ---
def process_video(video_path, output_json_path):
    """处理视频，提取姿态数据并保存为 JSON"""

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,        # 处理视频流
        model_complexity=1,             # 模型复杂度: 0, 1, 2 (越高越准但越慢)
        smooth_landmarks=True,          # 平滑关键点
        enable_segmentation=False,      # 不启用分割
        min_detection_confidence=0.5,   # 最低检测置信度
        min_tracking_confidence=0.5     # 最低跟踪置信度
    )
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info: FPS={fps:.2f}, Total Frames={frame_count}")

    all_frames_data = []
    frame_number = 0

    start_time = time.time()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("End of video or cannot read frame.")
            break

        # 计算当前时间戳
        timestamp = frame_number / fps

        # 将图像从 BGR 转换为 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False # 提高性能

        # 进行姿态检测
        results = pose.process(image_rgb)

        # 准备存储当前帧的数据
        frame_data = {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "landmarks": None,
            "angles": {}
        }

        # 如果检测到姿态
        if results.pose_landmarks:
            landmarks_list = []
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks_list.append({
                    "index": idx,
                    "name": mp_pose.PoseLandmark(idx).name,
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                })
            frame_data["landmarks"] = landmarks_list

            # 计算指定的角度
            landmarks = results.pose_landmarks.landmark
            for p1_idx, p2_idx, p3_idx, angle_name in ANGLES_TO_CALCULATE:
                try:
                    point1 = landmarks[p1_idx.value]
                    point2 = landmarks[p2_idx.value]
                    point3 = landmarks[p3_idx.value]

                    # 确保关节点可见性足够高 (可选，但推荐)
                    if point1.visibility > 0.5 and point2.visibility > 0.5 and point3.visibility > 0.5:
                        angle = calculate_angle(point1, point2, point3)
                        frame_data["angles"][angle_name] = angle
                    else:
                         frame_data["angles"][angle_name] = None # 或标记为不可靠
                except IndexError:
                    print(f"Warning: Landmark index out of bounds for angle {angle_name} in frame {frame_number}")
                    frame_data["angles"][angle_name] = None
                except Exception as e:
                    print(f"Error calculating angle {angle_name} in frame {frame_number}: {e}")
                    frame_data["angles"][angle_name] = None

            # --- 可选: 在图像上绘制姿态 ---
            # image.flags.writeable = True
            # image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # 转回 BGR 以便 OpenCV 显示
            # mp_drawing.draw_landmarks(
            #     image_bgr,
            #     results.pose_landmarks,
            #     mp_pose.POSE_CONNECTIONS,
            #     landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            #     connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            # )
            # cv2.imshow('MediaPipe Pose', image_bgr)
            # if cv2.waitKey(5) & 0xFF == 27: # 按 ESC 退出预览
            #     break
            # --- 可选预览结束 ---

        all_frames_data.append(frame_data)
        frame_number += 1

        # 打印进度
        if frame_number % 100 == 0:
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / frame_number) * frame_count if frame_number > 0 else 0
            print(f"Processed frame {frame_number}/{frame_count}. Elapsed: {elapsed_time:.2f}s. Estimated total: {estimated_total_time:.2f}s")


    # 释放资源
    cap.release()
    # cv2.destroyAllWindows() # 如果开启了预览，取消注释这行
    pose.close()

    # 保存数据到 JSON 文件
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_frames_data, f, ensure_ascii=False, indent=4)
        print(f"Successfully processed video and saved data to {output_json_path}")
    except IOError as e:
        print(f"Error writing JSON file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during JSON saving: {e}")


# --- 执行处理 ---
if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
    else:
        process_video(VIDEO_PATH, OUTPUT_JSON_PATH)
