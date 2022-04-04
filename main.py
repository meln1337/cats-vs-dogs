import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# Hyparameters
epochs = 10
lr = 3e-4
batch_size = 64
img_size = 128

transform = {
    'train': T.Compose([
        T.Resize([img_size, img_size]),
        T.ToTensor()
    ]),
    'test': T.Compose([
        T.Resize([img_size, img_size]),
        T.ToTensor()
    ])
}

train_dataset = ImageFolder(root='/files/cats-vs-dogs/train', transform=transform['train'])
train_classes = train_dataset.class_to_idx

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)