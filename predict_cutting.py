import os
import cv2
from ultralytics import YOLO

# 加载 YOLO 模型
model = YOLO("/home/xlsk/Code/jg-dx-rep/runs/detect/train3/weights/best.pt")

# 输入图像文件夹
img_folder = "/home/xlsk/Code/jg-dx-rep/my_dataset/train/uncovered"
# 输出裁剪图像文件夹
output_folder = "/home/xlsk/Code/jg-dx-rep/my_dataset/train/uncovered_crop"
os.makedirs(output_folder, exist_ok=True)

# 初始化序号
crop_index = 1

# 遍历文件夹中的所有图像文件
for img_name in os.listdir(img_folder):
    img_path = os.path.join(img_folder, img_name)
    img = cv2.imread(img_path)

    # 进行预测
    results = model(img)

    # 遍历所有预测框
    for result in results:
        # 获取预测框的坐标
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # 裁剪图像
            crop_img = img[y1:y2, x1:x2]

            # 保存裁剪后的图像，按序号命名
            crop_img_name = f"{crop_index}.jpg"
            crop_img_path = os.path.join(output_folder, crop_img_name)
            cv2.imwrite(crop_img_path, crop_img)

            # 增加序号
            crop_index += 1

print("裁剪图像保存完毕！")
