import torch
from torch import nn
from torch.nn import functional as F, parameter


class NEFBasedNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.para = parameter.Parameter(torch.ones(3),requires_grad=True)

    def forward(self, x, y):
        s = []
        for key in ['PER', 'ORG', 'LOC']:
            s.append(0)
            for a in x[key]:
                for b in y[key]:
                    if a[0] == b[0]:
                        s[-1] += a[1] * b[1]
        s = torch.tensor(s)
        s = s.dot(self.para)
        return s

