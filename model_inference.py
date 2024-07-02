import os
from ultralytics import YOLO
from tqdm import tqdm

model = YOLO("runs/detect/train/weights/best.pt")

img_folder = 'images_for_detection/images/val'  # 图片文件夹路径
img_names = [os.path.splitext(f)[0] for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]

for img_name in tqdm(img_names):
    img_path = os.path.join(img_folder, img_name + '.jpg')
    results = model(img_path)
    save_path = os.path.join("images_deteted", img_name + '.jpg')
    for result in results:
        result.save(filename=f"{save_path}")