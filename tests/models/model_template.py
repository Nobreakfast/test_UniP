import torch
import torch.nn as nn


class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, 1, 1)
        self.conv2 = nn.Conv2d(4, 4, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
