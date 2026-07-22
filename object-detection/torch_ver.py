import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
model.eval()
with torch.no_grad():
    prediction = model([torch.randn(3, 400, 600)])

print(prediction[0].keys())
print(f"boxes:  {prediction[0]['boxes'].shape}")
print(f"scores: {prediction[0]['scores'].shape}")
print(f"labels: {prediction[0]['labels'].shape}")