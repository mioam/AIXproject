import torch
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(768 * 2, 768),
        #     nn.ReLU(),
        #     nn.Linear(768, 768),
        #     nn.ReLU(),
        #     nn.Linear(768, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 2)
        # )
        self.net = nn.Sequential(
            nn.Linear(768 * 2, 768 * 2),
            nn.GLU(),
            nn.Linear(768, 768),
            nn.GLU(),
            nn.Linear(768 // 2, 2)
        )

    def forward(self, x, y):
        a = torch.cat((x,y), 1)
        a = self.net(a)
        return a
