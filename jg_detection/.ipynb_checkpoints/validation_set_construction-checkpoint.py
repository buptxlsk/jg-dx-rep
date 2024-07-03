import os
import shutil
import random

def split_dataset(img_folder, lbl_folder, train_img_folder, val_img_folder, train_lbl_folder, val_lbl_folder, split_ratio=0.9):
    # 获取所有图片文件名（不含扩展名）
    img_files = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]
    random.shuffle(img_files)
    
    # 计算训练集和验证集的大小
    train_size = int(len(img_files) * split_ratio)
    
    # 创建目标文件夹
    os.makedirs(train_img_folder, exist_ok=True)
    os.makedirs(val_img_folder, exist_ok=True)
    os.makedirs(train_lbl_folder, exist_ok=True)
    os.makedirs(val_lbl_folder, exist_ok=True)
    
    # 移动文件到训练集和验证集
    for i, img_file in enumerate(img_files):
        img_name, img_ext = os.path.splitext(img_file)
        lbl_file = img_name + '.txt'
        
        if i < train_size:
            shutil.move(os.path.join(img_folder, img_file), os.path.join(train_img_folder, img_file))
            shutil.move(os.path.join(lbl_folder, lbl_file), os.path.join(train_lbl_folder, lbl_file))
        else:
            shutil.move(os.path.join(img_folder, img_file), os.path.join(val_img_folder, img_file))
            shutil.move(os.path.join(lbl_folder, lbl_file), os.path.join(val_lbl_folder, lbl_file))
            
    print("数据集划分完成。")

# 示例用法
img_folder = './data/images'  # 图片文件夹路径
lbl_folder = './data/labels'  # 标签文件夹路径

train_img_folder = './data/images/train'  # 训练集图片文件夹路径
val_img_folder = './data/images/val'  # 验证集图片文件夹路径
train_lbl_folder = './data/labels/train'  # 训练集标签文件夹路径
val_lbl_folder = './data/labels/val'  # 验证集标签文件夹路径

split_dataset(img_folder, lbl_folder, train_img_folder, val_img_folder, train_lbl_folder, val_lbl_folder, split_ratio=0.9)
