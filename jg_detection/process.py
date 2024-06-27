import os
import shutil

def copy_matching_txt_files(img_folder, txt_folder, dest_folder):
    # 获取图片文件夹中的所有文件名（不含扩展名）
    img_names = [os.path.splitext(f)[0] for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]
    
    # 确保目标文件夹存在
    os.makedirs(dest_folder, exist_ok=True)
    
    # 遍历txt文件夹，找到名称匹配的txt文件并复制到目标文件夹
    for txt_file in os.listdir(txt_folder):
        txt_name, txt_ext = os.path.splitext(txt_file)
        if txt_ext.lower() == '.txt' and txt_name in img_names:
            src_path = os.path.join(txt_folder, txt_file)
            dest_path = os.path.join(dest_folder, txt_file)
            shutil.copy(src_path, dest_path)
            print(f'复制文件: {src_path} -> {dest_path}')

# 示例用法
img_folder = '/home/cxl/xwz/jg_detection/data/images'  # 图片文件夹路径
txt_folder = '/home/cxl/xwz/jg_detection/data/labels_notfine'  # txt文件夹路径
dest_folder = '/home/cxl/xwz/jg_detection/data/labels'  # 目标文件夹路径

copy_matching_txt_files(img_folder, txt_folder, dest_folder)
