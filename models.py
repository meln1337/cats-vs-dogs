import torch
import torch.nn as nn

class Classificator(nn.Module):
    def __init__(self, img_channels: int, img_size: int, features: int) -> None:
        super(Classificator, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.features = features
        self.classificator = nn.Sequential(
            # Input: (N, 3, 128, 128)
            self._block(img_channels, features, 3, 1, 1),
            # (N, F, 64, 64)
            self._block(features, features*2, 3, 1, 1),
            # (N, F*2, 32, 32)
            self._block(features*2, features*4, 3, 1, 1),
            # (N, F*4, 16, 16)
            self._block(features*4, features*8, 3, 1, 1),
            # (N, F*8, 8, 8)
            self._block(features*8, features*16, 3, 1, 1),
            # (N, F*16, 4, 4)
            nn.Conv2d(features*16, 1, 4, 1, 0),
            # (N, 1, 1, 1)
            nn.Flatten(),
            # (N, 1)
            nn.Sigmoid()
            # (N, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classificator(x)

    def _block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

def test():
    N, F, H, W = 64, 3, 128, 128
    x = torch.randn((N, F, H, W))
    classificator = Classificator(F, 128, 8)
    x = classificator(x)
    assert x.shape == (N, 1), "Test failed"