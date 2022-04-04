import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from models import Classificator
from utils import get_accuracy
from config import transform, batch_size, img_size, img_channels, features, device, lr, epochs

train_dataset = ImageFolder(root="/files/cats-vs-dogs/train", transform=transform["train"])
train_classes = train_dataset.class_to_idx
print(train_classes)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

classificator = Classificator(img_channels=img_channels, img_size=img_size, features=features).to(device)
optimizer = optim.Adam(classificator.parameters(), lr=lr)
critic = nn.BCELoss()

for epoch in range(epochs):
    loss_per_epoch = 0
    accuracy_per_epoch = 0
    train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch: {epoch+1}")
    for batch_idx, (X_train, Y_train) in train_loop:
        X_train = X_train.to(torch.float32).to(device)
        Y_train = Y_train.to(torch.float32).to(device)

        pred = classificator(X_train).reshape(-1) # (N, 1) => (N)
        loss = critic(pred, Y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            accuracy = get_accuracy(pred, Y_train)
            loss_per_epoch += loss.item()
            accuracy_per_epoch += accuracy.item()

        train_loop.set_postfix(average_loss=loss.item() * len(train_loop), average_accuracy=accuracy.item())

    print("")
    print(f"Epoch: {epoch+1}, Loss per epoch: {loss_per_epoch}, "
          f"Accuracy per epoch: {accuracy_per_epoch / len(train_dataloader)}")
    print("")

print("Training has been finished")
torch.save(classificator, "./classificator.pth")
print(f"The model of the classificator has been saved at: \"{os.path.join(os.path.dirname(__file__), 'classificator.pth')}\"")