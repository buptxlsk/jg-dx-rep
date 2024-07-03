import os
from PIL import Image

# 设置文件夹路径
image_folder = 'images'
output_folder = 'images/images_cut'

# 如果输出文件夹不存在，创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历images文件夹中的所有文件
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg'):
        image_path = os.path.join(image_folder, filename)
        txt_path = os.path.join(image_folder, filename.replace('.jpg', '.txt'))
        
        # 读取图像
        with Image.open(image_path) as img:
            width, height = img.size

            # 读取对应的txt文件
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.split()
                    class_id = int(parts[0])
                    if class_id == 0:
                        # 获取bounding box的中心坐标和宽高
                        x_center = float(parts[1]) * width
                        y_center = float(parts[2]) * height
                        bbox_width = float(parts[3]) * width
                        bbox_height = float(parts[4]) * height
                        
                        # 计算左上角和右下角的坐标
                        x1 = int(x_center - bbox_width / 2)
                        y1 = int(y_center - bbox_height / 2)
                        x2 = int(x_center + bbox_width / 2)
                        y2 = int(y_center + bbox_height / 2)
                        
                        # 裁剪图像
                        cropped_img = img.crop((x1, y1, x2, y2))
                        
                        # 保存裁剪后的图像
                        cropped_img.save(os.path.join(output_folder, filename))

print("图像裁剪完成。")
