import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.batch1(self.conv1(x)))
        x = F.relu(self.batch2(self.conv2(x)))
        return self.pool(x)
    
class multiKernelVGG(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.stack = nn.Sequential(
            VGG(3, 32),
            VGG(32, 64),
            VGG(64, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.stack(x))
    
vgg = multiKernelVGG()
x = torch.randn(1, 3, 32, 32)
print(f"o/p: {vgg(x).shape}")
print(f"params: {sum(p.numel() for p in vgg.parameters()):,}")