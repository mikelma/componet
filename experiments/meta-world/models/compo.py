import torch
import torch.nn as nn
import numpy as np

from .shared_arch import shared

import sys, os

sys.path.append(os.path.dirname(__file__) + "/../../../")
from componet import CompoNet, FirstModuleWrapper


def net(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, output_dim),
    )


def take_first(x):
    return x[0]


class CompoNetAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, prev_paths, map_location=None):
        super().__init__()

        self.net_logstd = net(obs_dim, act_dim)

        prev_units = [
            FirstModuleWrapper(
                model=torch.load(
                    f"{prev_paths[0]}/model.pt", map_location=map_location
                ),
                ret_probs=False,
                transform_output=take_first,
            )
        ]
        prev_units += [
            torch.load(f"{p}/net_mean.pt", map_location=map_location)
            for p in prev_paths[1:]
        ]

        self.net_mean = CompoNet(
            previous_units=prev_units,
            input_dim=obs_dim,
            hidden_dim=256,
            out_dim=act_dim,
            internal_policy=net(obs_dim + 256, act_dim),
            ret_probs=False,
        )

    def forward(self, x, writer=None, global_step=None):
        if writer is None or global_step is None:
            mean = self.net_mean(x)[0]
        else:
            mean, _phi, att_in, att_out, int_pol, head_out = self.net_mean(
                x, return_atts=True, ret_int_pol=True, ret_head_out=True
            )
            # log attention values
            for i, v in enumerate(att_in.mean(0)[0].detach()):
                writer.add_scalar(f"charts/att_in_{i}", v.item(), global_step)
            for i, v in enumerate(att_out.mean(0)[0].detach()):
                writer.add_scalar(f"charts/att_out_{i}", v.item(), global_step)

            with torch.no_grad():
                dist_out_int_pol = (mean - int_pol).abs().mean().item()
                dist_out_head_out = (mean - head_out).abs().mean().item()
                dist_int_pol_head_out = (int_pol - head_out).abs().mean().item()

                writer.add_scalar(
                    f"charts/dist_out_int_pol", dist_out_int_pol, global_step
                )
                writer.add_scalar(
                    f"charts/dist_out_head_out", dist_out_head_out, global_step
                )
                writer.add_scalar(
                    f"charts/dist_int_pol_head_out", dist_int_pol_head_out, global_step
                )

        logstd = self.net_logstd(x)
        return mean, logstd

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        del self.net_mean.previous_units
        torch.save(self.net_mean, f"{dirname}/net_mean.pt")
        torch.save(self.net_logstd, f"{dirname}/net_logstd.pt")

    def load(dirname, obs_dim, act_dim, prev_paths, map_location=None):
        print("Loading previous:", prevs_paths)

        model = CompoNetAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            prevs_paths=prevs_paths,
            map_location=map_location,
        )
        model.net_logstd = torch.load(
            f"{dirname}/net_logstd.pt", map_location=map_location
        )

        net_mean = torch.load(f"{dirname}/net_logstd.pt", map_location=map_location)

        curr = model.net_mean.state_dict()
        other = net_mean.state_dict()
        for k in other:
            curr[k] = other[k]
        model.net_mean.load_state_dict(curr)

        return model
