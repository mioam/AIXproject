import torch
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self, num_layer=2, act='ReLU', hidden_size=768, feature_type=0) -> None:
        super().__init__()
        self.feature_type = feature_type
        
        in_size = 768 * ([2,3,1][feature_type])
        hidden_size_in = hidden_size // 2 if act == 'GLU' else hidden_size
        Act = nn.ReLU if act == 'ReLU' else nn.GLU

        self.net = [nn.Linear(in_size, hidden_size),
                Act(),
            ] 
        for i in range(num_layer - 2):
            self.net.extend([
                nn.Linear(hidden_size_in, hidden_size),
                Act(),
            ])
        self.net.extend([nn.Linear(hidden_size_in, 2)])
        self.net = nn.Sequential(*self.net)

    def forward(self, x, y):
        if self.feature_type == 0:
            a = torch.cat((x,y), 1)
        elif self.feature_type == 1:
            a = torch.cat((x,y,x*y), 1)
        elif self.feature_type == 2:
            a = x*y
        a = self.net(a)
        return a

class AttnNet(nn.Module):
    def __init__(self, x_dim, y_dim, num_heads) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.linear = nn.Linear(768, x_dim * y_dim)
        self.attn = nn.MultiheadAttention(y_dim, num_heads, batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(x_dim * y_dim,768),
            nn.ReLU(),
            nn.Linear(768,2),
        )

    def forward(self, x, y):
        x = self.linear(x).reshape(-1, self.x_dim, self.y_dim)
        y = self.linear(y).reshape(-1, self.x_dim, self.y_dim)
        a = self.attn(x,y,y)[0].reshape(-1, self.x_dim * self.y_dim)
        a = self.net(a)
        return a
