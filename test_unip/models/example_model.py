import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np
from einops import rearrange


class ExampleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.p1 = nn.Parameter(torch.randn(1, 1, 1, 4))
        self.conv2 = nn.Conv2d(4, 8, 3, 1, 1)
        self.conv3 = nn.Conv2d(4, 8, 3, 1, 1)
        self.bn23 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, 1, groups=16)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5_1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.identity = nn.Identity()
        self.bn5i = nn.BatchNorm2d(16)
        self.bn45 = nn.BatchNorm2d(32)
        self.fcp = nn.Linear(32, 32)
        self.conv6 = nn.Conv2d(32, 1, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(1)
        self.p6 = nn.Parameter(torch.randn(1, 1, 1, 1))
        # self.pool = nn.MaxPool2d(2, 2)
        self.pool = nn.AvgPool2d(2, 2)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = x + self.p1
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.conv2(x1)
        x2 = self.conv3(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.bn23(x)
        identity = self.identity(x)
        x1 = self.conv4(x)
        x1 = self.bn4(x1)
        x2 = self.conv5_1(x)
        x2 = self.conv5_2(x2 + identity)
        x2 = self.bn5i(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.bn45(x)  # (1, 32, 4, 4)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.fcp(x)  # (1, 4, 4, 32) -> (1, 4, 4, 32)
        x = rearrange(x, "b (h w) c -> b c h w", h=4)
        x = self.conv6(x)  # (1, 32, 4, 4)
        x = self.bn6(x)
        x = torch.nn.functional.relu(x)
        x = x * self.p6
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc(x)
        return x
