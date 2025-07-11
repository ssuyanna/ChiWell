import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import json

# 配置参数
model_path = 'medicine_model.pth'
image_path = '/Users/suya/Desktop/test.jpg'  # 待识别图片
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载类别名
with open("class_names.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)
num_classes = len(class_names)

# 构建模型
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 加载并处理图片
img = Image.open(image_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0).to(device)

# 执行预测
with torch.no_grad():
    output = model(input_tensor)
    predicted_idx = torch.argmax(output, dim=1).item()
    predicted_class = class_names[predicted_idx]

print(f"预测类别为：{predicted_class}")