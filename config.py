import torch
import torchvision.transforms as T

# Hyparameters
epochs = 5
lr = 1e-3
batch_size = 64
img_size = 128
img_channels = 3
test_img_num = 12500
features = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = {
    'train': T.Compose([
        T.Resize([img_size, img_size]),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.ToTensor()
    ]),
    'test': T.Compose([
        T.Resize([img_size, img_size]),
        T.ToTensor()
    ])
}