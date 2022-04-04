import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from config import batch_size, device, transform, test_img_num

res = torch.zeros(test_img_num)

test_dataset = ImageFolder("/files/cats-vs-dogs/test", transform=transform["test"])
test_dataloader = DataLoader(batch_size=batch_size, dataset=test_dataset, shuffle=False)

classificator = torch.load("./classificator.pth", map_location=device)

i = 0
with torch.no_grad():
    for (X_train, _) in test_dataloader:
        X_train = X_train.to(torch.float32).to(device)
        local_batch_size = X_train.shape[0]
        pred = classificator(X_train).reshape(-1)
        res[i:i+local_batch_size] = pred.to("cpu")
        i += local_batch_size

res = res.to("cpu").numpy()

df = pd.DataFrame({
    "id": range(1, test_img_num+1),
    "label": res
})
df.to_csv("./submission.csv", index=False)
print("Submission has been saved")