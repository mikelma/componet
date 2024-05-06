import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from .dino_encoder import DinoEncoder


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DinoSimpleAgent(nn.Module):
    def __init__(self, envs, dino_size, frame_stack, device):
        super().__init__()
        self.dino = DinoEncoder(dino_size=dino_size, device=device)
        self.middle = nn.Sequential(
            layer_init(nn.Linear(self.dino.embed_dim * frame_stack, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        h = self.middle(self.dino.encode(x * 255.0))  # denormalize
        return self.critic(h)

    def get_action_and_value(self, x, action=None):
        hidden = self.middle(self.dino.encode(x * 255.0))  # denormalize
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def save(self, dirname, model_name):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.middle, f"{dirname}/{model_name}_middle.pt")
        torch.save(self.actor, f"{dirname}/{model_name}_actor.pt")
        torch.save(self.critic, f"{dirname}/{model_name}_critic.pt")

    def load(dirname, agent_args):
        model = DinoSimpleAgent(**agent_args)
        model.middle = torch.load(f"{dirname}/middle.pt")
        model.actor = torch.load(f"{dirname}/actor.pt")
        model.critic = torch.load(f"{dirname}/critic.pt")
        return model
