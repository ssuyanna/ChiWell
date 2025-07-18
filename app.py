import os
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, Response
from werkzeug.utils import secure_filename
import json
import io  # 用于处理内存中的图片
import uuid  # 用于生成唯一文件名

# --- 中草药识别相关导入 ---
import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import torchvision.models as models
import torch.nn as nn
# --- 结束导入 ---

# --- 八段锦动作识别相关导入 ---
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from model import LSTMClassifier  # 从 model.py 导入

# --- 结束导入 ---
from flask import send_file  # 新增导入
UPLOAD_FOLDER = '/tmp/uploads'

HERB_IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, 'herb_images')
ACTION_VIDEO_FOLDER = os.path.join(UPLOAD_FOLDER, 'action_videos')  # 存放上传的动作视频
ACTION_JSON_FOLDER = os.path.join(UPLOAD_FOLDER, 'action_json')  # 存放提取的姿态JSON
ACUPOINTS_DATA_FOLDER = 'acupoints_data'  # 存储穴位数据的文件夹

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'wmv'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HERB_IMAGE_FOLDER, exist_ok=True)
os.makedirs(ACTION_VIDEO_FOLDER, exist_ok=True)
os.makedirs(ACTION_JSON_FOLDER, exist_ok=True)
os.makedirs(ACUPOINTS_DATA_FOLDER, exist_ok=True)  # 创建穴位数据文件夹

# --- 中草药识别模型加载 ---
MEDICINE_MODEL_PATH = 'medicine_model.pth'
CLASS_NAMES_PATH = 'class_names.json'
herb_model = None
herb_class_names = []
herb_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
herb_transform = None
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"


import timm                                       # ← 新增
from torchvision import transforms, models        # 旧行保留

# ……

# ---------- herb model ----------
MEDICINE_MODEL_PATH = "medicine_model.pth"
CLASS_NAMES_PATH = "class_names.json"
def load_herb_model():
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    model = timm.create_model("convnext_tiny.fb_in22k",
                              pretrained=False,
                              num_classes=len(class_names))
    ckpt = torch.load(MEDICINE_MODEL_PATH, map_location=herb_device)
    model.load_state_dict(ckpt.get("state", ckpt), strict=False)
    model.to(herb_device).eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    return model, transform, class_names

# !!! 别忘了把返回值赋给全局变量
herb_model, herb_transform, herb_class_names = load_herb_model()

# --- 八段锦动作识别模型加载 ---
ACTION_MODEL_PATH = "./output/model.pt"  # 你的 LSTM 模型路径
action_model = None
action_device = torch.device('cpu')  # LSTM 通常在 CPU 上也很快

# 从 predict.py 借鉴的常量
ACTION_NUM_CLASSES = 10
ACTION_MAX_FRAMES = 60
ACTION_ANGLE_KEYS = [
    "left_elbow_angle", "right_elbow_angle",
    "left_knee_angle", "right_knee_angle",
    "left_hip_angle", "right_hip_angle"
]
ACTION_ID_TO_LABEL = {
    0: "两手托天理三焦", 1: "左右开弓似射雕", 2: "调理脾胃须单举",
    3: "五劳七伤往后瞧", 4: "摇头摆尾去心火", 5: "两手攀足固肾腰",
    6: "攒拳怒目增气力", 7: "背后七颠百病消", 8: "预备式", 9: "收势"
}


def load_action_recognition_model():
    global action_model
    print("--- Loading Baduanjin Action Recognition Model ---")
    try:
        if not os.path.exists(ACTION_MODEL_PATH):
            raise FileNotFoundError(f"Action recognition model file not found: {ACTION_MODEL_PATH}")

        action_model = LSTMClassifier(input_dim=len(ACTION_ANGLE_KEYS), num_classes=ACTION_NUM_CLASSES)
        action_model.load_state_dict(torch.load(ACTION_MODEL_PATH, map_location=action_device))
        action_model.to(action_device)
        action_model.eval()
        print(f"Action recognition model loaded successfully onto {action_device}.")
        print("--- Action Model Loading Complete ---")
    except FileNotFoundError as e:
        print(f"[ERROR] Loading action recognition model failed: {e}. Action recognition feature will be disabled.")
        action_model = None
    except Exception as e:
        import traceback
        print(f"[ERROR] An unexpected error occurred during action model loading: {type(e).__name__} - {e}")
        print(traceback.format_exc())
        action_model = None
        print("--- Action Model Loading Failed ---")


load_action_recognition_model()
# --- 结束模型加载 ---
# --- 全局变量存储穴位数据 ---
acupoints_db = {}  # { "经脉显示名": { "穴位显示名": {"id": "meridian_folder/point_folder", "image_filename": "image.jpg", "description": "text"} } }
acupoint_id_map = {}  # { "meridian_folder/point_folder": {"meridian": "经脉显示名", "point": "穴位显示名"} }


def generate_display_name(folder_name, is_meridian=False):
    """从文件夹名称生成更易读的显示名称。"""
    # 移除可能存在的前导/尾随空格或特殊字符，尽管os.listdir通常不会产生这些
    clean_folder_name = folder_name.strip()
    parts = clean_folder_name.split('_')
    if not parts:
        return clean_folder_name

    if is_meridian:
        chinese_name_parts = [p for p in parts if any('\u4e00' <= char <= '\u9fff' for char in p)]
        pinyin_parts = [p for p in parts if not any('\u4e00' <= char <= '\u9fff' for char in p)]

        chinese_name = "".join(chinese_name_parts) if chinese_name_parts else " ".join(parts)  # Fallback
        pinyin_name = "_".join(pinyin_parts)

        if pinyin_name and chinese_name != pinyin_name:  # 避免 "手太阴肺经 (手太阴肺经)"
            # 确保拼音部分不只是中文部分的重复
            if not any(p_part in chinese_name for p_part in pinyin_parts):
                return f"{chinese_name} ({pinyin_name})"
        return chinese_name

    else:  # 例如 "LU-1_Zhongfu_中府" -> "中府 Zhongfu (LU-1)"
        code = parts[0]
        name_elements = []
        chinese_elements = []
        for part in parts[1:]:
            if any('\u4e00' <= char <= '\u9fff' for char in part):
                chinese_elements.append(part)
            else:
                name_elements.append(part)

        display_parts = []
        if chinese_elements:
            display_parts.append("".join(chinese_elements))
        if name_elements:  # 确保英文名在中文名之后，如果都有
            display_parts.append(" ".join(name_elements).capitalize())

        name_str = " ".join(display_parts)
        return f"{name_str} ({code})" if name_str else code


def load_acupoints_data():
    global acupoints_db, acupoint_id_map
    print("--- Loading Acupoints Data ---")
    acupoints_data_temp = {}
    id_map_temp = {}

    if not os.path.exists(ACUPOINTS_DATA_FOLDER):
        print(f"[WARN] Acupoints data folder not found: {ACUPOINTS_DATA_FOLDER}")
        return

    for meridian_folder_name in sorted(os.listdir(ACUPOINTS_DATA_FOLDER)):
        meridian_path = os.path.join(ACUPOINTS_DATA_FOLDER, meridian_folder_name)
        # 确保是目录且不是隐藏文件/文件夹
        if os.path.isdir(meridian_path) and not meridian_folder_name.startswith('.'):
            meridian_display_name = generate_display_name(meridian_folder_name, is_meridian=True)
            acupoints_data_temp[meridian_display_name] = {}
            print(f"  Loading Meridian: {meridian_folder_name} -> Display: '{meridian_display_name}'")

            for point_folder_name in sorted(os.listdir(meridian_path)):
                point_path = os.path.join(meridian_path, point_folder_name)
                if os.path.isdir(point_path) and not point_folder_name.startswith('.'):
                    point_display_name = generate_display_name(point_folder_name)
                    # 使用原始文件夹名构建ID，因为这些用于路径查找
                    point_id = f"{meridian_folder_name}/{point_folder_name}"

                    image_file = None
                    description_text = "Description not found for this acupoint."  # Default text

                    for item in os.listdir(point_path):
                        if item.lower().startswith("image.") and item.lower().split('.')[
                            -1] in ALLOWED_IMAGE_EXTENSIONS:
                            image_file = item
                            break

                    desc_file_path = os.path.join(point_path, "text.txt")
                    if os.path.exists(desc_file_path):
                        try:
                            with open(desc_file_path, 'r', encoding='utf-8') as f_desc:
                                description_text = f_desc.read().strip()
                                if not description_text:  # Handle empty description file
                                    description_text = "Description is available but currently empty."
                        except Exception as e:
                            print(f"[ERROR] Reading description for {point_id}: {e}")
                            description_text = "Error reading description file."

                    acupoints_data_temp[meridian_display_name][point_display_name] = {
                        "id": point_id,  # This ID is crucial for fetching
                        "image_filename": image_file,
                        "description": description_text
                    }
                    id_map_temp[point_id] = {"meridian": meridian_display_name, "point": point_display_name}
                    # print(f"    Loaded Acupoint: {point_folder_name} -> '{point_display_name}' (ID: {point_id}, Image: {image_file})")

    acupoints_db = acupoints_data_temp
    acupoint_id_map = id_map_temp
    if acupoints_db:
        print(f"--- Acupoints Data Loading Complete. Found {len(acupoints_db)} meridians. ---")
        # Optional: Print a sample for debugging
        # first_meridian_key = next(iter(acupoints_db))
        # print(f"Sample from '{first_meridian_key}': {json.dumps(list(acupoints_db[first_meridian_key].items())[:2], ensure_ascii=False, indent=2)}")
    else:
        print(f"--- Acupoints Data Loading Complete. No data found. ---")


# ... (之前的 load_herb_model, load_action_recognition_model) ...
load_acupoints_data()  # 确保在模型加载后或独立加载


# ... (allowed_file 和其他路由) ...


def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in allowed_extensions


# --- 路由定义 ---
@app.route('/favicon.ico')
def favicon():
    return Response(status=204)


@app.route('/')
def index():
    # 默认页面是练习页
    return render_template('practice.html', active_page='practice')


@app.route('/practice')
def practice_page():
    return render_template('practice.html', active_page='practice')


@app.route('/upload_analysis')
def upload_page():
    return render_template('upload.html', active_page='upload')


@app.route('/herb_identifier')
def herb_identifier_page():
    return render_template('herb_identifier.html', active_page='herb_identifier')


@app.route('/get_standard_data')
def get_standard_data():
    try:
        filepath = 'standard_pose_data.json'
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Standard pose data file not found at: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list) or not data:
            raise ValueError("Standard pose data is empty or not a list.")
        return jsonify(data)
    except Exception as e:
        print(f"[ERROR] /get_standard_data: {type(e).__name__}: {e}")
        return jsonify({"error": "Failed to load standard data."}), 500


# --- API: 处理动作视频上传、姿态提取和预测 ---
# =============================================================================
# 请复制下面的完整函数，并替换掉 app.py 中旧的 predict_action_route 函数
# =============================================================================
# =============================================================================
# 请复制下面的完整函数，并替换掉 app.py 中旧的 predict_action_route 函数
# =============================================================================
@app.route('/predict_action', methods=['POST'])
def predict_action_route():
    if 'video' not in request.files:
        return jsonify({"error": "No video file part"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        try:
            # 1. 保存用户上传的视频
            unique_id = str(uuid.uuid4())
            original_filename, ext = os.path.splitext(file.filename)
            secure_name = secure_filename(original_filename)
            video_filename = f"{secure_name}_{unique_id}{ext}"
            video_path = os.path.join(ACTION_VIDEO_FOLDER, video_filename)
            file.save(video_path)
            print(f"User video saved to: {video_path}")

            # 2. 配置大语言模型
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.4,  # 可以稍微提高一点温度，让语言生动些
            )

            # =====================================================================================
            # 这是最终版本的 Prompt，请用它完整替换 app.py 中已有的 prompt 变量
            # =====================================================================================
            prompt = PromptTemplate.from_template("""
            你是一位顶级的运动康复与健康顾问以及经验老道的中医，擅长从生物力学结合中医的角度分析动作，并给出精准、易懂的指导。

            **核心指令 (最重要):**
            你的任务是生成一份关于用户动作的改进建议报告。在你的报告中，**绝对不要尝试识别或提及用户正在练习的八段锦动作的具体名称**。你的所有建议都应聚焦于身体姿态、动作质量和改进方法，而不是动作的命名。

            **分析流程:**
            1.  **观察用户动作**: 仔细观看【用户练习视频】中的每一个细节。
            2.  **寻找参考**: 在【标准动作视频】中找到与用户动作形态最相似的片段作为你的内在分析参考。
            3.  **生成通用性建议**: 基于上述的内在比较，从以下方面生成一份**不提及具体动作名称**的通用性评估报告：
                - **身体姿态与排列**: 用户的脊柱是否中立？头部、肩部、髋部的位置关系是否理想？是否存在不必要的紧张？
                - **动作质量**: 动作的幅度是否充分？发力是否流畅？整个动作的节奏控制得如何？
                - **核心改进建议**: 给出1-3条最关键、最普适的改进点。例如：“建议在向上伸展时，更专注于核心收紧，以保护腰部”或“你的下蹲幅度可以再稍大一些，同时确保膝盖与脚尖方向一致”。

            **输入路径:**
            - 标准动作视频: {standard_path}
            - 用户练习视频: {user_path}

            **输出要求 (必须严格遵守):**
            - **返回一个单一的、连续的文本字符串。**
            - **不要使用JSON或任何代码块格式。**
            - 报告中**严禁出现任何八段锦招式名称**（如“两手托天理三焦”等）。
            - 报告包含两部分：先是完整的中文报告，然后是完整的英文报告。
            - 使用 "--- English Report ---" 作为中英文报告的分隔符。

            **输出格式示例:**

            【中文部分】
            你好！这是为你准备的动作练习报告。通过分析你的动作，我们发现了一些可以让你做得更好的地方：

            1.  **关于身体姿态**: 我们注意到你的肩部在手臂上举时有轻微的耸起。下次可以尝试有意识地让肩膀下沉，这能更好地打开胸腔。
            2.  **核心改进建议**: 在整个动作过程中，请更专注于保持核心区域的稳定，这会让你下盘更稳，发力也更顺畅。

            --- English Report ---

            Hello! Here is the practice report for your exercise. After analyzing your movements, we found some areas for improvement:

            1.  **Regarding Posture**: We noticed a slight shrugging of the shoulders as you raised your arms. Next time, try to consciously keep your shoulders down to better open up your chest.
            2.  **Core Improvement Tip**: Throughout the movement, please focus more on maintaining a stable core. This will give you better balance and smoother power generation.
            """)
            # 4. [已更新] 构建调用链，解析器改回 StrOutputParser
            # 因为我们现在需要的是一个包含中英文的完整字符串
            parser = StrOutputParser()
            chain = prompt | llm | parser

            # 5. [已更新] 执行调用链，返回的结果现在是一个单一的字符串
            report = chain.invoke({
                "standard_path": os.path.abspath("static/videos/baduanjin.mp4"),
                "user_path": os.path.abspath(video_path)
            })

            # 6. 直接返回模型生成的完整双语报告字符串
            return jsonify({
                "report": report,  # report里现在包含了中英文两部分
                "message": "动作分析成功!"
            }), 200

        except Exception as e:
            import traceback
            print(f"[ERROR] Gemini 分析失败: {e}")
            print(traceback.format_exc())
            return jsonify({"error": "Gemini 分析过程中发生错误"}), 500

    else:
        return jsonify({"error": "不支持的视频格式"}), 400


# --- 处理草药图片上传和预测的 API ---
@app.route('/predict_herb', methods=['POST'])
def predict_herb():
    if herb_model is None or herb_transform is None or not herb_class_names:
        print("[ERROR] /predict_herb: Attempted prediction while model is not loaded.")
        return jsonify(
            {"error": "Herb recognition model is not loaded or failed to load, function is not available."}), 503

    if 'image' not in request.files:
        return jsonify({"error": "No image file part"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        try:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            input_tensor = herb_transform(img).unsqueeze(0).to(herb_device)

            with torch.no_grad():
                output = herb_model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

            predicted_class = herb_class_names[predicted_idx.item()]

            print(f"Predicted herb: {predicted_class} for file: {file.filename}")
            return jsonify({
                "prediction": predicted_class,
            }), 200
        except UnidentifiedImageError:
            print(f"[ERROR] /predict_herb: Cannot identify image file format for {file.filename}")
            return jsonify({"error": "Unrecognised image file format."}), 400
        except Exception as e:
            import traceback
            print(
                f"[ERROR] /predict_herb: Error processing image or predicting for {file.filename}: {type(e).__name__} - {e}")
            print(traceback.format_exc())
            return jsonify({"error": "An internal error occurred while processing an image or prediction."}), 500
    else:
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'N/A'
        print(f"Upload rejected: Disallowed image file type - {file_ext}")
        return jsonify({"error": f"Disallowed image file types: .{file_ext}"}), 400


# --- 新增穴位查询相关路由 ---

@app.route('/acupoints')
def acupoints_page():
    return render_template('acupoints.html', active_page='acupoints', meridians_data=acupoints_db)


@app.route('/acupoint_data/<path:point_id>')
def get_acupoint_data(point_id):
    # print(f"API: Request for acupoint_data with ID: '{point_id}'") # Debugging
    # print(f"API: Current acupoint_id_map keys: {list(acupoint_id_map.keys())[:5]}") # Debugging
    if point_id in acupoint_id_map:
        meridian_display_name = acupoint_id_map[point_id]["meridian"]
        point_display_name = acupoint_id_map[point_id]["point"]
        data = acupoints_db.get(meridian_display_name, {}).get(point_display_name)
        if data:
            image_url = None
            if data.get("image_filename"):
                # Ensure point_id for url_for is the same used in the route definition
                image_url = url_for('get_acupoint_image', point_id=data["id"])  # Use the stored ID

            # print(f"API: Found data for {point_id}: Name: {point_display_name}, Image URL: {image_url}") # Debugging
            return jsonify({
                "name": point_display_name,  # Use display name for consistency
                "meridian": meridian_display_name,  # Use display name
                "description": data.get("description", "No description available."),
                "image_url": image_url
            })
        else:
            # print(f"API: Data not found in acupoints_db for {meridian_display_name} / {point_display_name}") # Debugging
            return jsonify({"error": f"Acupoint data details not found for ID {point_id} after map lookup."}), 404
    # print(f"API: Point ID '{point_id}' not found in acupoint_id_map.") # Debugging
    return jsonify({"error": f"Acupoint ID '{point_id}' not found."}), 404


@app.route('/acupoint_image/<path:point_id>')
def get_acupoint_image(point_id):
    # print(f"API: Request for acupoint_image with ID: '{point_id}'") # Debugging
    if point_id in acupoint_id_map:
        # The point_id IS "meridian_folder/point_folder"
        meridian_folder, point_folder = point_id.split('/', 1)

        # Retrieve display names to access acupoints_db, then get the actual image_filename
        meridian_display_name = acupoint_id_map[point_id]["meridian"]
        point_display_name = acupoint_id_map[point_id]["point"]

        image_filename = acupoints_db.get(meridian_display_name, {}).get(point_display_name, {}).get("image_filename")

        if image_filename:
            # Construct path using original folder names (which are in point_id)
            image_path = os.path.join(ACUPOINTS_DATA_FOLDER, meridian_folder, point_folder, image_filename)
            # print(f"API: Attempting to send image from path: {image_path}") # Debugging
            if os.path.exists(image_path):
                return send_file(image_path)
            else:
                # print(f"API: Image file not found at path: {image_path}") # Debugging
                return jsonify({"error": "Image file physically not found on server."}), 404
        else:
            # print(f"API: No image_filename recorded for point ID '{point_id}'") # Debugging
            return jsonify({"error": "No image filename associated with this acupoint."}), 404
    # print(f"API: Point ID '{point_id}' not found in map for image request.") # Debugging
    return jsonify({"error": "Acupoint ID not found for image."}), 404


# --- 更新导航栏的 active_page 传递 ---
# 第一个 @app.route('/') 定义已经存在于大约第 231 行，所以下面的重复定义被移除了。
# @app.route('/')
# def index():
#     # 默认页面可以是练习页或穴位查询页，这里假设是练习页
#     return render_template('practice.html', active_page='practice') # 或者 'acupoints'

# 确保其他页面路由也传递正确的 active_page (这些看起来是注释掉的示例，可以保持原样或根据需要启用)
# @app.route('/practice')
# def practice_page():
#     return render_template('practice.html', active_page='practice')

# @app.route('/upload_analysis')
# def upload_page():
#     return render_template('upload.html', active_page='upload')

# @app.route('/herb_identifier')
# def herb_identifier_page():
#     return render_template('herb_identifier.html', active_page='herb_identifier')

# [新增] 全新的、纯 API 调用的聊天机器人接口
@app.route('/ask_chatbot', methods=['POST'])
def ask_chatbot():
    """
    处理来自前端悬浮聊天窗口的请求。
    新增了对聊天记录的处理，实现了记忆功能。
    """
    # 1. 从前端请求中获取用户的问题和聊天记录
    data = request.get_json()
    question = data.get("question")
    # history 是一个列表，包含之前的对话，例如:
    # [{"role": "user", "content": "你好"}, {"role": "ai", "content": "你好！"}]
    history = data.get("history", [])

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    try:
        # 2. 初始化 Gemini 2.5 Flash 模型 (与之前相同)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.6,
            convert_system_message_to_human=True
        )

        # 3. [修改] 创建一个支持聊天记录的 Prompt
        # 我们使用 MessagesPlaceholder 来为历史消息创建一个“占位符”
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一位精通中医养生智慧、态度友善的健康助手。请根据上下文直接回答用户提出的最新问题。请用清晰、易懂、鼓励性的语言进行交流。"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        # 4. 构建并执行调用链 (与之前类似)
        chain = prompt | llm | StrOutputParser()

        # 5. [新增] 将前端传来的 history 转换成 LangChain 能理解的格式
        chat_history_for_chain = []
        for msg in history:
            if msg.get("role") == "user":
                chat_history_for_chain.append(HumanMessage(content=msg.get("content")))
            elif msg.get("role") == "ai":
                chat_history_for_chain.append(AIMessage(content=msg.get("content")))

        # 6. [修改] 调用链，并传入格式化后的聊天记录和新问题
        answer = chain.invoke({
            "chat_history": chat_history_for_chain,
            "question": question
        })

        # 7. [新增] 更新聊天记录，并确保它不超过10条
        history.append({"role": "user", "content": question})
        history.append({"role": "ai", "content": answer})

        # 如果历史记录超过10条，就只保留最新的10条
        if len(history) > 10:
            history = history[-10:]

        # 8. [修改] 将 AI 的回答和更新后的历史记录一并返回给前端
        return jsonify({"answer": answer, "history": history})

    except Exception as e:
        import traceback
        print(f"[ERROR] Chatbot API failed: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "An error occurred while calling the AI service."}), 500


if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)