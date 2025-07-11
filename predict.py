import torch
import torch.nn.functional as F
import json
import numpy as np
from model import LSTMClassifier

# ✅ 模型与标签设置
MODEL_PATH = "./output/model.pt"
INPUT_JSON = "data/processed_json/section_5.json"
NUM_CLASSES = 10
MAX_FRAMES = 60
ANGLE_KEYS = [
    "left_elbow_angle", "right_elbow_angle",
    "left_knee_angle", "right_knee_angle",
    "left_hip_angle", "right_hip_angle"
]

# ✅ 可选：类别 ID → 中文映射（如需修改可替换）
ID_TO_LABEL = {
    0: "两手托天理三焦",
    1: "左右开弓似射雕",
    2: "调理脾胃须单举",
    3: "五劳七伤往后瞧",
    4: "摇头摆尾去心火",
    5: "两手攀足固肾腰",
    6: "攒拳怒目增气力",
    7: "背后七颠百病消",
    8: "预备式",
    9: "收势"
}


def load_json_to_tensor(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        frames = json.load(f)

    angle_seq = []
    for frame in frames:
        angles = frame.get("angles", {})
        vec = [angles.get(k, 0.0) for k in ANGLE_KEYS]
        angle_seq.append(vec)

    angle_seq = np.array(angle_seq)

    # 补齐/截断为 [MAX_FRAMES, 6]
    T, D = angle_seq.shape
    if T < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - T, D))
        angle_seq = np.vstack([angle_seq, pad])
    elif T > MAX_FRAMES:
        idx = np.linspace(0, T - 1, MAX_FRAMES).astype(int)
        angle_seq = angle_seq[idx]

    tensor = torch.tensor(angle_seq, dtype=torch.float32).unsqueeze(0)  # shape: [1, T, 6]
    return tensor


def predict(model, input_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        return pred_class, probs.squeeze().tolist()


def main():
    # ✅ 加载模型
    model = LSTMClassifier(input_dim=6, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

    # ✅ 处理输入 JSON
    input_tensor = load_json_to_tensor(INPUT_JSON)

    # ✅ 模型预测
    pred_id, prob = predict(model, input_tensor)
    pred_label = ID_TO_LABEL.get(pred_id, f"动作 {pred_id}")

    print(f"\n✅ 预测结果：类别编号 = {pred_id}")
    print(f"📌 动作名称：{pred_label}")
    print(f"🔢 预测概率分布（前几个类）: {prob[:5]}")


if __name__ == "__main__":
    main()
