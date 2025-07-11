# filepath: /Users/suya/Desktop/Baduanjin_711/clean_filenames.py
import os
import re

def sanitize_filename(filename):
    """
    Removes special characters that can cause issues with Git and web servers.
    Keeps original extension.
    """
    # 分离文件名和扩展名
    name, ext = os.path.splitext(filename)
    
    # 移除所有非字母、数字、下划线、短横线或点的字符
    # 保留中文字符
    sanitized_name = re.sub(r'[^\w\s\-\.\u4e00-\u9fa5]', '_', name)
    
    # 替换空格为下划线
    sanitized_name = re.sub(r'\s+', '_', sanitized_name)
    
    # 避免多个下划线
    sanitized_name = re.sub(r'__+', '_', sanitized_name)
    
    # 重新组合文件名和扩展名
    return sanitized_name + ext

def clean_directory(root_dir):
    """
    Recursively walks through a directory and renames files and folders.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # 重命名文件
        for filename in filenames:
            if filename == '.DS_Store' or filename == 'clean_filenames.py':
                continue
            original_path = os.path.join(dirpath, filename)
            sanitized = sanitize_filename(filename)
            new_path = os.path.join(dirpath, sanitized)
            if original_path != new_path:
                print(f'Renaming file: "{original_path}" -> "{new_path}"')
                os.rename(original_path, new_path)
        
        # 重命名文件夹
        for dirname in dirnames:
            original_path = os.path.join(dirpath, dirname)
            sanitized = sanitize_filename(dirname)
            new_path = os.path.join(dirpath, sanitized)
            if original_path != new_path:
                print(f'Renaming directory: "{original_path}" -> "{new_path}"')
                os.rename(original_path, new_path)

if __name__ == "__main__":
    # 从当前脚本所在的位置开始清理
    project_root = os.path.dirname(os.path.abspath(__file__))
    print(f"Starting cleanup in: {project_root}")
    # 我们只清理数据和图片目录，避免误伤其他目录
    target_dirs = ['herbal_medicine', 'acupoints_data', 'static']
    for directory in target_dirs:
        path_to_clean = os.path.join(project_root, directory)
        if os.path.exists(path_to_clean):
            print(f"\n--- Cleaning {directory} ---")
            clean_directory(path_to_clean)
        else:
            print(f"\n--- Directory {directory} not found, skipping. ---")
    print("\nCleanup complete!")