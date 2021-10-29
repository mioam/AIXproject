import torch
from torch import nn
from torch.nn import functional as F


class BertNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.postnet = nn.Sequential(
            nn.Linear(768, 768)
        )

    def forward(self, x):
        # print(x.shape, y.shape)
        x = self.postnet(x)
        x = x / x.norm()
        return x

# 未完成


