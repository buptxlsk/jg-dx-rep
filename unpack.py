import rarfile
import os

def extract_rar(rar_file_path, dest_folder):
    # 确保目标文件夹存在
    os.makedirs(dest_folder, exist_ok=True)
    
    with rarfile.RarFile(rar_file_path) as rf:
        rf.extractall(dest_folder)
        print(f'解压文件: {rar_file_path} 到 {dest_folder}')

# 示例用法
rar_file_path = '/home/dx/usrs/xwz/jg_detection.rar'  # RAR文件路径
dest_folder = '/home/dx/usrs/xwz'  # 解压目标文件夹路径

extract_rar(rar_file_path, dest_folder)
