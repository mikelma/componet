import torch
import torch.nn as nn
import os


class ProgressiveNetAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, prev_paths=[], map_location=None):
        super().__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim

        if len(prev_paths) > 0:
            prevs = [
                torch.load(f"{p}/prognet.pt", map_location=map_location)
                for p in prev_paths
            ]
        else:
            prevs = []
        self.fc = ProgressiveNet(
            input_dim=obs_dim,
            hidden_dim=256,
            previous_models=prevs,
        )

        # will be created when calling `reset_heads`
        self.fc_mean = None
        self.fc_logstd = None
        self.reset_heads()

    def reset_heads(self):
        self.fc_mean = nn.Linear(256, self.act_dim)
        self.fc_logstd = nn.Linear(256, self.act_dim)

    def forward(self, x, global_timestep=None):
        x = self.fc(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        return mean, log_std

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        if hasattr(self.fc, "previous_models"):
            del self.fc.previous_models
        torch.save(self.fc, f"{dirname}/prognet.pt")
        torch.save(self.fc_mean, f"{dirname}/fc_mean.pt")
        torch.save(self.fc_logstd, f"{dirname}/fc_logstd.pt")

    def load(
        dirname, obs_dim, act_dim, prev_paths, map_location=None, reset_heads=False
    ):
        model = ProgressiveNetAgent(obs_dim, act_dim, prev_paths, map_location)
        model.fc = torch.load(f"{dirname}/model.pt", map_location=map_location)
        if reset_heads:
            return model
        else:
            model.fc_mean = torch.load(
                f"{dirname}/fc_mean.pt", map_location=map_location
            )
            model.fc_logstd = torch.load(
                f"{dirname}/fc_logstd.pt", map_location=map_location
            )
        return model


class ProgressiveNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, previous_models=[]):
        super().__init__()

        n_prev = len(previous_models)
        self.n_prev = n_prev

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.u2 = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_prev)]
        )

        self.a = nn.ReLU()

        # freeze previous models (columns)
        for m in previous_models:
            if hasattr(m, "previous_models"):
                del m.previous_models
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        self.previous_models = previous_models

    def forward_first(self, x):
        fc1 = self.a(self.fc1(x))
        out = self.a(self.fc2(fc1))
        return out, fc1

    def forward_other(self, x, fc1s):
        fc1 = self.a(self.fc1(x))

        assert len(fc1s) == len(
            self.u2
        ), "The number of previous layer outputs does not match the number of adapters"
        h2 = [u(fc1s[i]) for i, u in enumerate(self.u2)]
        fc2 = sum([self.fc2(fc1), *h2])
        out = self.a(fc2)

        return out, fc1

    def forward(self, x, prev_hs=dict(fc1s=[])):
        if len(self.previous_models) == 0:
            return self.forward_first(x)[0]

        # forward first module
        _, fc1 = self.forward_first(x)

        fc1s = [fc1]
        for model in self.previous_models[1:]:
            _, fc1 = model.forward_other(x, fc1s)
            fc1s.append(fc1)
        out = self.forward_other(x, fc1s)[0]
        return out
