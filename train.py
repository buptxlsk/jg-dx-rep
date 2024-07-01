import torch
from ultralytics import YOLO

# 检查CUDA是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# 打印所使用的设备
print(f'Using device: {device}')

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from YAML
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="jg.yaml", epochs=100, device=device)
