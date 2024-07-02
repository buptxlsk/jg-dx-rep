import os
import json
from PIL import Image

# 定义文件夹路径
images_folder = 'images'

# 遍历文件夹中的所有json文件
for file_name in os.listdir(images_folder):
    if file_name.endswith('.json'):
        json_path = os.path.join(images_folder, file_name)
        
        # 读取json文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取图片的宽度和高度
        image_file = data['imagePath']
        image_path = os.path.join(images_folder, image_file)
        
        # 使用PIL获取图像尺寸
        image = Image.open(image_path)
        image_width, image_height = image.size
        
        # 获取标注信息
        shapes = data['shapes']
        
        # 写入到同名的txt文件中
        txt_file_name = file_name.replace('.json', '.txt')
        txt_file_path = os.path.join(images_folder, txt_file_name)
        
        with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
            # 遍历所有标注
            for shape in shapes:
                label = shape['label']
                points = shape['points']
                
                # YOLO标签
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                # 转换为YOLO格式
                x_center = (x1 + x2) / 2 / image_width
                y_center = (y1 + y2) / 2 / image_height
                width = abs(x2 - x1) / image_width
                height = abs(y2 - y1) / image_height
                
                # 写入标签
                yolo_label = f'{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n'
                txt_file.write(yolo_label)
