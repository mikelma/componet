import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
import os


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ProgressiveNetAgent(nn.Module):
    def __init__(self, envs, prevs_paths=[], map_location=None):
        super().__init__()
        hidden_dim = 512

        prevs = [
            torch.load(f"{p}/encoder.pt", map_location=map_location)
            for p in prevs_paths
        ]

        self.encoder = ProgressiveNet(
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            previous_models=prevs,
            layer_init=layer_init,
        )

        self.actor = layer_init(
            nn.Linear(hidden_dim, envs.single_action_space.n), std=0.01
        )
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1)

    def get_value(self, x):
        hidden = self.encoder(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self.encoder(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        if hasattr(self.encoder, "previous_models"):
            del self.encoder.previous_models
        torch.save(self.actor, f"{dirname}/actor.pt")
        torch.save(self.encoder, f"{dirname}/encoder.pt")
        torch.save(self.critic, f"{dirname}/crititc.pt")

    def load(dirname, envs, prevs_paths, map_location=None):
        model = ProgressiveNetAgent(envs=envs, prevs_paths=prevs_paths)
        model.actor = torch.load(f"{dirname}/actor.pt", map_location=map_location)
        model.encoder = torch.load(f"{dirname}/encoder.pt", map_location=map_location)
        model.critic = torch.load(f"{dirname}/critic.pt", map_location=map_location)
        return model


class ProgressiveNet(nn.Module):
    def __init__(
        self, hidden_dim, out_dim, previous_models=[], layer_init=lambda x, **kwargs: x
    ):
        super().__init__()

        n_prev = len(previous_models)
        self.n_prev = n_prev

        self.c1 = layer_init(nn.Conv2d(4, 32, 8, stride=4))

        self.c2 = layer_init(nn.Conv2d(32, 64, 4, stride=2))
        self.u2 = nn.ModuleList(
            [
                layer_init(nn.Conv2d(32, 64, 4, stride=2, bias=False))
                for _ in range(n_prev)
            ]
        )

        self.c3 = layer_init(nn.Conv2d(64, 64, 3, stride=1))
        self.u3 = nn.ModuleList(
            [
                layer_init(nn.Conv2d(64, 64, 3, stride=1, bias=False))
                for _ in range(n_prev)
            ]
        )

        self.a = nn.ReLU()
        self.flat = nn.Flatten()

        self.o1 = layer_init(nn.Linear(64 * 7 * 7, out_dim))
        self.uo1 = nn.ModuleList(
            [
                layer_init(nn.Linear(64 * 7 * 7, out_dim, bias=False))
                for _ in range(n_prev)
            ]
        )

        # freeze previous models (columns)
        for m in previous_models:
            if hasattr(m, "previous_models"):
                del m.previous_models
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        self.previous_models = previous_models

    def forward_first(self, x):
        c1 = self.c1(x)
        c2 = self.c2(self.a(c1))
        c3 = self.c3(self.a(c2))

        out = self.o1(self.a(self.flat(c3)))

        return out, c1, c2, c3

    def forward_other(self, x, c1s, c2s, c3s):
        c1 = self.a(self.c1(x))

        assert len(c1s) == len(
            self.u2
        ), "The number of previous layer outputs does not match the number of adapters"

        h2 = [u(c1s[i]) for i, u in enumerate(self.u2)]
        c2 = sum([self.c2(c1), *h2])
        c2 = self.a(c2)

        h3 = [u(c2s[i]) for i, u in enumerate(self.u3)]
        c3 = sum([self.c3(c2), *h3])
        c3 = self.a(c3)

        uo1 = [u(self.flat(c3s[i])) for i, u in enumerate(self.uo1)]
        out = sum([self.o1(self.flat(c3)), *uo1])

        return out, c1, c2, c3

    def forward(self, x):
        if len(self.previous_models) == 0:
            return self.forward_first(x)[0]

        # forward first module
        _, c1, c2, c3 = self.forward_first(x)

        # forward the rest of the previous modules (if there're some)
        c1s, c2s, c3s = [c1], [c2], [c3]
        for model in self.previous_models[1:]:
            _, c1, c2, c3 = model.forward_other(x, c1s, c2s, c3s)
            c1s.append(c1)
            c2s.append(c2)
            c3s.append(c3)
        out, _, _, _ = self.forward_other(x, c1s, c2s, c3s)
        return out
