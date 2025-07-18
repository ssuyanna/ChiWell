import torch, json
import timm                                 # ← 新增
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# 配置参数
model_path = 'models/convnext_herb_best.pth' # 新权重
image_path = '/Users/suya/Desktop/test.jpg'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载类别名
with open('models/class_names.json', 'r', encoding='utf-8') as f:
    class_names = json.load(f)
num_classes = len(class_names)

# 构建模型（把 resnet18 改成 convnext_tiny.fb_in22k）
model = timm.create_model(
    'convnext_tiny.fb_in22k',
    pretrained=False,
    num_classes=num_classes
)
ckpt = torch.load(model_path, map_location=device)
if 'state' in ckpt:              # 训练脚本里包了一层
    ckpt = ckpt['state']
model.load_state_dict(ckpt, strict=False)
model = model.to(device)
model.eval()

# 图像预处理（保持与训练 eval_tf 一致）
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

# 加载并处理图片
img = Image.open(image_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0).to(device)

# 执行预测 (Top-1)
with torch.no_grad():
    output = model(input_tensor)
    predicted_idx = torch.argmax(output, dim=1).item()
    predicted_class = class_names[predicted_idx]

print(f"The predicted category is {predicted_class}")   # 原行保留
