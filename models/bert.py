import torch
from torch import nn
from torch.nn import functional as F


class BertNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.postnet = nn.Sequential(
            nn.Linear(768, 768)
        )

    def forward(self, x, y):
        # print(x.shape, y.shape)
        x = self.postnet(x.reshape(1,-1)).reshape(-1)
        y = self.postnet(y.reshape(1,-1)).reshape(-1)
        x = x / x.norm()
        y = y / y.norm()
        return (x * y).sum()

# 未完成


