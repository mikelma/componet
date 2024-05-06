import os
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from .cnn_encoder import CnnEncoder
import sys, os

sys.path.append(os.path.dirname(__file__) + "/../../../")
from componet import CompoNet, FirstModuleWrapper


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CnnCompoNetAgent(nn.Module):
    def __init__(self, envs, prevs_paths=[], finetune_encoder=False, map_location=None):
        super().__init__()
        hidden_dim = 512

        if not finetune_encoder or len(prevs_paths) == 0:
            self.encoder = CnnEncoder(hidden_dim=hidden_dim, layer_init=layer_init)
        else:
            self.encoder = torch.load(
                f"{prevs_paths[-1]}/encoder.pt", map_location=map_location
            )
            print("==> Encoder loaded from last CompoNet module")

        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1)

        pol_in = hidden_dim if len(prevs_paths) == 0 else hidden_dim * 2
        pol = nn.Sequential(
            layer_init(nn.Linear(pol_in, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, envs.single_action_space.n), std=0.01),
        )

        if len(prevs_paths) > 0:
            previous_units = [
                torch.load(f"{p}/actor.pt", map_location=map_location)
                for p in prevs_paths
            ]
            self.actor = CompoNet(
                previous_units=previous_units,
                input_dim=hidden_dim,  # size of the CnnEncoder output
                hidden_dim=hidden_dim,
                out_dim=envs.single_action_space.n,
                internal_policy=pol,
                ret_probs=True,
                encoder=self.encoder,
            )
            self.is_compo = True
        else:
            self.actor = FirstModuleWrapper(
                model=pol, ret_probs=True, encoder=self.encoder
            )
            self.is_compo = False

    def get_value(self, x):
        return self.critic(self.encoder(x))

    def get_action_and_value(
        self, x, action=None, log_writter=None, global_step=None, prevs_to_noise=0
    ):
        if not self.is_compo or global_step is None or log_writter is None:
            p, _phi, hidden = self.actor(
                x, ret_encoder_out=True, prevs_to_noise=prevs_to_noise
            )
        else:
            p, _phi, hidden, att_in, att_out, int_pol, head_out = self.actor(
                x,
                ret_encoder_out=True,
                return_atts=True,
                ret_int_pol=True,
                ret_head_out=True,
                prevs_to_noise=prevs_to_noise,
            )

            with torch.no_grad():
                # log attention values
                for i, v in enumerate(att_in.mean(0)[0].detach()):
                    log_writter.add_scalar(f"charts/att_in_{i}", v.item(), global_step)
                for i, v in enumerate(att_out.mean(0)[0].detach()):
                    log_writter.add_scalar(f"charts/att_out_{i}", v.item(), global_step)

                _, a_int_pol = int_pol.max(-1)
                _, a_head_out = head_out.max(-1)
                _, a_out = p.max(-1)

                bs = a_out.size(0)
                log_writter.add_scalar(
                    f"charts/out_matches_int_pol",
                    ((a_out == a_int_pol).sum() / bs).item(),
                    global_step,
                )
                log_writter.add_scalar(
                    f"charts/out_matches_head_out",
                    ((a_out == a_head_out).sum() / bs).item(),
                    global_step,
                )
                log_writter.add_scalar(
                    f"charts/int_pol_matches_head",
                    ((a_int_pol == a_head_out).sum() / bs).item(),
                    global_step,
                )

        probs = Categorical(probs=p)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        if hasattr(self.actor, "previous_units"):
            del self.actor.previous_units
        torch.save(self.actor, f"{dirname}/actor.pt")
        torch.save(self.critic, f"{dirname}/crititc.pt")
        torch.save(self.encoder, f"{dirname}/encoder.pt")

    def load(dirname, envs, prevs_paths=[], map_location=None):
        print("Loading previous:", prevs_paths)

        model = CnnCompoNetAgent(
            envs=envs, prevs_paths=prevs_paths, map_location=map_location
        )
        model.encoder = torch.load(f"{dirname}/encoder.pt", map_location=map_location)

        # load the state dict of the actor
        actor = torch.load(f"{dirname}/actor.pt", map_location=map_location)
        curr = model.actor.state_dict()
        other = actor.state_dict()
        for k in other:
            curr[k] = other[k]
        model.actor.load_state_dict(curr)

        model.critic = torch.load(f"{dirname}/crititc.pt", map_location=map_location)
        return model
