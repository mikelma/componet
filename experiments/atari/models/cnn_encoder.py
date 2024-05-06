import torch.nn as nn


class CnnEncoder(nn.Module):
    def __init__(self, hidden_dim=512, layer_init=lambda x, **kwargs: x):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, hidden_dim)),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)
