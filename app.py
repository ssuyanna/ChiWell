import os
import subprocess
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, Response
from werkzeug.utils import secure_filename
import json
import io # 用于处理内存中的图片

# --- 中草药识别相关导入 ---
import torch
from torchvision import transforms
# 从 PIL 导入 Image 和 UnidentifiedImageError
from PIL import Image, UnidentifiedImageError
import torchvision.models as models
import torch.nn as nn
# --- 结束导入 ---

UPLOAD_FOLDER = 'uploads'
HERB_IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, 'herb_images') # 单独存放草药图片
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'wmv'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'} # 允许的图片类型

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # 保持视频大小限制

# --- 创建上传目录 ---
# 使用 os.makedirs(exist_ok=True) 更简洁
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HERB_IMAGE_FOLDER, exist_ok=True)
# --- 结束创建目录 ---

# --- 中草药识别模型加载 ---
MEDICINE_MODEL_PATH = 'medicine_model.pth'
CLASS_NAMES_PATH = 'class_names.json'
herb_model = None
herb_class_names = []
herb_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
herb_transform = None

def load_herb_model():
    global herb_model, herb_class_names, herb_transform
    print("--- Loading Herb Identification Model ---")
    try:
        # 检查模型和类别文件是否存在
        if not os.path.exists(MEDICINE_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MEDICINE_MODEL_PATH}")
        if not os.path.exists(CLASS_NAMES_PATH):
            raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

        # 加载类别名
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            herb_class_names = json.load(f)
        num_classes = len(herb_class_names)
        if num_classes == 0:
             raise ValueError("Class names file is empty or invalid.")
        print(f"Loaded {num_classes} herb class names.")

        # 构建模型结构 (确保与训练时一致)
        herb_model = models.resnet18(weights=None) # 加载结构，不加载预训练权重
        herb_model.fc = nn.Linear(herb_model.fc.in_features, num_classes)
        print("Model structure created (ResNet18).")

        # 加载你训练/微调后的模型权重
        herb_model.load_state_dict(torch.load(MEDICINE_MODEL_PATH, map_location=herb_device))
        herb_model = herb_model.to(herb_device)
        herb_model.eval() # 设置为评估模式
        print(f"Herb identification model weights loaded successfully onto {herb_device}.")

        # 定义图像预处理 (!!! 与你的训练代码保持一致 !!!)
        herb_transform = transforms.Compose([
            transforms.Resize((224, 224)), # 调整图像大小
            transforms.ToTensor()          # 转换为 Tensor (值范围 [0, 1])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 更新日志信息以反映实际的转换
        print("Herb image transform defined (Resize, ToTensor only).")
        print("--- Herb Model Loading Complete ---")

    except FileNotFoundError as e:
        print(f"[ERROR] Loading herb model failed: {e}. Herb identification feature will be disabled.")
        herb_model = None # 标记模型加载失败
    except Exception as e:
        # 打印更详细的错误信息，包括异常类型
        import traceback
        print(f"[ERROR] An unexpected error occurred during herb model loading: {type(e).__name__} - {e}")
        print(traceback.format_exc()) # 打印完整的堆栈跟踪
        herb_model = None
        print("--- Herb Model Loading Failed ---")


# 在应用启动时加载模型
load_herb_model()
# --- 结束模型加载 ---


def allowed_file(filename, allowed_extensions):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

# --- 路由定义 ---

@app.route('/favicon.ico')
def favicon():
    """处理浏览器图标请求"""
    return Response(status=204)

@app.route('/')
def index():
    """主页，重定向到练习页面"""
    # 确保传递 active_page
    return render_template('practice.html', active_page='practice')

@app.route('/practice')
def practice_page():
    """实时跟练页面"""
    return render_template('practice.html', active_page='practice')

@app.route('/upload_analysis')
def upload_page():
    """上传分析页面"""
    return render_template('upload.html', active_page='upload')

# --- 草药识别页面路由 ---
@app.route('/herb_identifier')
def herb_identifier_page():
    """草药识别页面"""
    # 传递 active_page 以便导航栏高亮
    return render_template('herb_identifier.html', active_page='herb_identifier')
# --- 结束草药识别页面路由 ---

@app.route('/get_standard_data')
def get_standard_data():
    """API: 提供标准姿态数据"""
    try:
        # 确保文件路径正确
        filepath = 'standard_pose_data.json'
        if not os.path.exists(filepath):
             raise FileNotFoundError(f"Standard pose data file not found at: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 简单的验证，确保数据是列表且不为空
        if not isinstance(data, list) or not data:
             raise ValueError("Standard pose data is empty or not a list.")
        return jsonify(data)
    except FileNotFoundError as e:
        print(f"[ERROR] /get_standard_data: {e}")
        return jsonify({"error": "Standard pose data file not found."}), 404
    except json.JSONDecodeError as e:
        print(f"[ERROR] /get_standard_data: Invalid JSON format - {e}")
        return jsonify({"error": "Invalid standard pose data format."}), 500
    except ValueError as e:
        print(f"[ERROR] /get_standard_data: Invalid data content - {e}")
        return jsonify({"error": f"Invalid standard pose data content: {e}"}), 500
    except Exception as e:
        print(f"[ERROR] /get_standard_data: Unexpected error - {type(e).__name__}: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/upload', methods=['POST'])
def upload_video():
    """API: 处理视频上传"""
    if 'video' not in request.files:
        return jsonify({"error": "No video file part"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            print(f"Video saved to: {filepath}")
            # 这里应该触发实际的后台分析任务
            return jsonify({"message": f"Video '{filename}' uploaded successfully. Backend analysis has started (simulation)."}), 200
        except Exception as e:
            print(f"[ERROR] Error saving video file '{filename}': {e}")
            return jsonify({"error": "Failed to save video file."}), 500
    else:
        # 提供更具体的文件类型错误信息
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'N/A'
        print(f"Upload rejected: Disallowed video file type - {file_ext}")
        return jsonify({"error": f"Disallowed video file types: .{file_ext}"}), 400

# --- 处理草药图片上传和预测的 API ---
@app.route('/predict_herb', methods=['POST'])
def predict_herb():
    """API: 处理草药图片上传和预测"""
    # 检查模型是否成功加载
    if herb_model is None or herb_transform is None or not herb_class_names:
        print("[ERROR] /predict_herb: Attempted prediction while model is not loaded.")
        return jsonify({"error": "Herb recognition model is not loaded or failed to load, function is not available."}), 503 # 503 Service Unavailable

    if 'image' not in request.files:
        return jsonify({"error": "No image file part"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        try:
            # 直接在内存中处理图片
            img_bytes = file.read()
            # 使用 Pillow 打开图像数据
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB') # 确保是 RGB 格式

            # --- (可选) 保存上传的图片 ---
            # filename_secure = secure_filename(file.filename)
            # save_path = os.path.join(HERB_IMAGE_FOLDER, filename_secure)
            # try:
            #     with open(save_path, 'wb') as f_save:
            #          f_save.write(img_bytes)
            #     print(f"Herb image saved to: {save_path}")
            # except Exception as save_err:
            #     print(f"[WARN] Could not save uploaded herb image '{filename_secure}': {save_err}")
            # --- 结束保存 ---

            # 图像预处理 (现在与训练代码一致)
            input_tensor = herb_transform(img).unsqueeze(0).to(herb_device) # 添加 batch 维度并移动到设备

            # 执行预测
            with torch.no_grad(): # 关闭梯度计算以节省内存和加速
                output = herb_model(input_tensor)
                probabilities = torch.softmax(output, dim=1) # 计算概率
                confidence, predicted_idx = torch.max(probabilities, 1) # 获取最高概率及其索引

            predicted_class = herb_class_names[predicted_idx.item()] # 获取预测的类别名称
            confidence_score = confidence.item() * 100 # 转换为百分比

            
            print(f"Predicted herb: {predicted_class} with confidence: {confidence_score:.2f}% for file: {file.filename}")
            
            # 返回预测结果和置信度
            return jsonify({
                "prediction": predicted_class,
                # "confidence": f"{confidence_score:.2f}%"
            }), 200

        except UnidentifiedImageError:
             # Pillow 无法识别图像格式
             print(f"[ERROR] /predict_herb: Cannot identify image file format for {file.filename}")
             return jsonify({"error": "Unrecognised image file format."}), 400
        except Exception as e:
            # 捕获其他所有潜在错误
            import traceback
            print(f"[ERROR] /predict_herb: Error processing image or predicting for {file.filename}: {type(e).__name__} - {e}")
            print(traceback.format_exc())
            return jsonify({"error": "An internal error occurred while processing an image or prediction."}), 500
    else:
        # 提供更具体的文件类型错误信息
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'N/A'
        print(f"Upload rejected: Disallowed image file type - {file_ext}")
        return jsonify({"error": f"Disallowed image file types: .{file_ext}"}), 400
# --- 结束 API ---


if __name__ == '__main__':
    print("Starting Flask application...")
    # host='0.0.0.0' 允许局域网访问
    # debug=True 会在代码更改时自动重载，并提供更详细的错误页面
    # use_reloader=False 可以防止模型被加载两次（在 debug 模式下）
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
    # 注意：如果 use_reloader=False 导致其他自动重载功能失效，
    # 你可能需要手动重启服务器来看代码更改。
    # 或者，将模型加载逻辑移到一个只执行一次的地方（例如使用 Flask 的 before_first_request 装饰器，但在较新版本中已弃用，推荐使用启动脚本或 WSGI 服务器配置）。
    # 对于简单的开发，use_reloader=False 通常是加载模型最直接的方式。