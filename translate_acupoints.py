# filepath: /Users/suya/Desktop/Baduanjin_711/translate_acupoints.py
import os
import time
from googletrans import Translator, LANGUAGES

# --- 配置 ---
# 要处理的根文件夹
ROOT_DIR = 'acupoints_data_translated'

# 创建一个翻译器实例
translator = Translator()

def translate_with_retry(text, dest='en', src='zh-cn', retries=3, delay=1):
    """
    带重试机制的翻译函数，以增加成功率。
    """
    for i in range(retries):
        try:
            # 避免翻译已经包含英文括号的内容
            if ' (' in text and ')' in text:
                return text
            translated = translator.translate(text, dest=dest, src=src)
            return translated.text
        except Exception as e:
            print(f"   - 翻译失败 (尝试 {i+1}/{retries}): {e}. 等待 {delay} 秒后重试...")
            time.sleep(delay)
    print(f"   - 警告: 翻译 '{text}' 最终失败，将使用原文本。")
    return text

def process_text_file(file_path):
    """
    处理单个 text.txt 文件，在每行中文后追加英文翻译。
    """
    print(f" - 正在处理文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 检查是否已经翻译过
        if ' (' in line and ')' in line:
            new_lines.append(line)
            continue

        parts = line.split('：', 1)
        if len(parts) == 2:
            key, value = parts
            print(f"   - 正在翻译: {value}")
            translated_value = translate_with_retry(value)
            new_line = f"{key}：{value} ({translated_value})"
            new_lines.append(new_line)
        else:
            new_lines.append(line) # 如果格式不符，保留原样

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    print(f"   - 文件更新完毕。")

def process_directories():
    """
    从下到上遍历目录，处理文件并重命名文件夹。
    """
    if not os.path.exists(ROOT_DIR):
        print(f"错误: 目录 '{ROOT_DIR}' 不存在。")
        return

    # 使用 topdown=False 从最深层目录开始处理
    for dirpath, dirnames, filenames in os.walk(ROOT_DIR, topdown=False):
        # 1. 处理当前目录下的 text.txt 文件
        if 'text.txt' in filenames:
            process_text_file(os.path.join(dirpath, 'text.txt'))

        # 2. 重命名当前目录下的子文件夹
        for dirname in dirnames:
            # 检查文件夹是否已经重命名过
            if ' (' in dirname and ')' in dirname:
                continue

            print(f"\n- 正在重命名文件夹: {dirname}")
            translated_dirname = translate_with_retry(dirname)
            new_dirname = f"{dirname} ({translated_dirname})"
            
            original_path = os.path.join(dirpath, dirname)
            new_path = os.path.join(dirpath, new_dirname)
            
            try:
                os.rename(original_path, new_path)
                print(f"  - 重命名成功: '{dirname}' -> '{new_dirname}'")
            except OSError as e:
                print(f"  - 错误: 重命名文件夹 '{dirname}' 失败: {e}")
        
        # 短暂延时，避免请求过于频繁
        time.sleep(0.5)

if __name__ == '__main__':
    print("--- 开始批量翻译和重命名 acupoints_data ---")
    process_directories()
    print("\n--- 所有操作完成！ ---")
