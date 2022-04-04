import torch

def get_accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.where(pred >= 0.5, 1, 0)
    return (pred == target).sum() / pred.shape[0]

# print(get_accuracy(torch.tensor([0, 0, 1, 0]), torch.tensor([0, 0, 1, 1])))