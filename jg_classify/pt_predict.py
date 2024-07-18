import os
import shutil
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

def create_unique_folder(base_name):
    folder_num = 1
    while os.path.exists(f'{base_name}{folder_num}'):
        folder_num += 1
    return f'{base_name}{folder_num}'

# 数据转换（与训练时相同）
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 定义数据集路径和预测路径
new_data_dir = 'new_data'
output_base_dir = 'predict'

# 创建唯一的预测输出文件夹
output_dir = create_unique_folder(output_base_dir)
os.makedirs(output_dir)

# 加载类别名称
class_names = ['broke', 'circle', 'good', 'lose', 'uncovered'] 

# 加载模型结构和权重
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
#model.fc = nn.Linear(num_ftrs, len(class_names))
model.fc = nn.Sequential(
  nn.Linear(num_ftrs, 1024),
  nn.ReLU(),
  nn.Dropout(0.5),
  nn.Linear(1024, 512),
  nn.ReLU(),
  nn.Dropout(0.5),
  nn.Linear(512, len(class_names))
)
model.load_state_dict(torch.load('./train2/best_model.pt'))
model.eval()

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 读取图像并进行预测
def predict_image(image_path, model, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 增加批次维度
    image = image.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    
    return preds.item()

# 预测并组织文件
for img_file in os.listdir(new_data_dir):
    img_path = os.path.join(new_data_dir, img_file)
    if os.path.isfile(img_path):
        pred_class_idx = predict_image(img_path, model, data_transforms)
        pred_class_name = class_names[pred_class_idx]
        
        # 创建目标文件夹
        predicted_folder = os.path.join(output_dir, f"{pred_class_name}_predict")
        if not os.path.exists(predicted_folder):
            os.makedirs(predicted_folder)
        
        # 移动文件到对应的预测文件夹
        destination_path = os.path.join(predicted_folder, img_file)
        shutil.copy(img_path, destination_path)

print("Prediction and organization complete.")
