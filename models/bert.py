import torch
from torch import nn
from torch.nn import functional as F


class BertBasedNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.postnet = nn.Sequential(
            nn.Linear(768, 768)
        )

    def forward(self, x, y):
        x = self.postnet(x)
        y = self.postnet(y)
        return (x * y).sum()

# 未完成

