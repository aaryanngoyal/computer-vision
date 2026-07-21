# torchvision wraps the heavy components

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, Normalize, ToTensor

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train = Compose([
    RandomCrop(32, padding=4, padding_mode="reflect"),
    RandomHorizontalFlip(), 
    ToTensor(),
    Normalize(mean, std),
])

eval = Compose([ToTensor, Normalize(mean, std)])

train_ds = CIFAR10(root="./data", train=True,  download=True, transform=train)
val_ds   = CIFAR10(root="./data", train=False, download=True, transform=eval)