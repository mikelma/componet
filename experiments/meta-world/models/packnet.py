import torch
import torch.nn as nn
import copy
import os
import numpy as np


class PackNetAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, task_id, total_task_num, device):
        super().__init__()

        shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.network = PackNet(
            model=shared,
            task_id=task_id + 1,
            total_task_num=total_task_num,
            is_first_task=task_id == 0,
            device=device,
        )
        self.num_actions = act_dim
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)

        self.retrain_mode = False

    def forward(self, x):
        h = self.network(x)
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        return mean, log_std

    def save(self, dirname):
        # un-do the masking for the current task
        self.network.set_view(None)

        os.makedirs(dirname, exist_ok=True)
        torch.save(self, f"{dirname}/packnet.pt")

    def load(
        dirname, task_id=None, restart_heads=False, freeze_bias=True, map_location=None
    ):
        model = torch.load(f"{dirname}/packnet.pt", map_location=map_location)
        model.retrain_mode = False

        if task_id is not None:
            model.network.task_id = task_id

        if restart_heads:
            model.fc_mean = nn.Linear(256, model.num_actions)
            model.fc_logstd = nn.Linear(256, model.num_actions)

        if freeze_bias:
            for name, param in model.network.model.named_parameters():
                if name.endswith(".bias"):
                    param.requires_grad = False
        return model

    def start_retraining(self):
        if self.retrain_mode:
            return  # nothing to do

        print("==> PackNet re-training starts!")

        self.retrain_mode = True
        self.network.prune()  # generate the masks for the current task
        self.network.set_view(self.network.task_id)

    def before_update(self):
        self.network.adjust_gradients(retrain_mode=self.retrain_mode)


class PackNet(nn.Module):
    def __init__(
        self,
        model,
        task_id,
        total_task_num,
        is_first_task,
        layer_init=lambda x, **kwargs: x,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        assert task_id > 0, "Task ID must be greater than 0 in PackNet"

        self.model = model
        self.task_id = task_id
        self.total_task_num = total_task_num
        self.prune_percentage = 1 / self.total_task_num

        self.view = None
        self.handled_layers = (
            []
        )  # will contain copies of the original parameters when using views

        # generate the masks
        self.masks = []
        for name, param in self.model.named_parameters():
            if name.endswith(".weight"):
                self.masks.append(
                    torch.zeros(param.size(), dtype=torch.long, device=device)
                )
            else:
                self.masks.append(None)

        # if we're in a tasks that it's not the first, freeze biases
        if not is_first_task:
            for name, param in self.model.named_parameters():
                if name.endswith(".bias"):
                    param.requires_grad = False

    @torch.no_grad()
    def adjust_gradients(self, retrain_mode=False):
        mask_id = self.task_id if retrain_mode else 0
        for p, mask in zip(self.model.parameters(), self.masks):
            if mask is None:
                continue
            p.grad = p.grad * (mask == mask_id)

    @torch.no_grad()
    def prune(self):
        for p, mask in zip(self.model.parameters(), self.masks):
            if mask is None:
                continue

            # sort the unassigned weights from lower to higher magnitudes
            masked = p * (mask == 0)  # only select "free" weights
            flat = masked.flatten()
            _sorted, indices = torch.sort(
                flat.abs(), descending=True
            )  # sort from max to min magnitude
            n_prune = int(
                self.prune_percentage * flat.size(0)
            )  # number of weights to keep in pruning

            # create the mask
            mask.flatten()[indices[:n_prune]] = self.task_id

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def set_view(self, task_id):
        if task_id is None and self.view is not None:
            # restore the original state of the model in the free parameters (not masked)
            for param_copy, param, mask in zip(
                self.handled_layers, self.model.parameters(), self.masks
            ):
                if param_copy is None:
                    continue
                m = torch.logical_and(
                    mask <= self.view, mask > 0
                )  # pruned=0, not-pruned=1
                param.data += param_copy.data * torch.logical_not(m)

            self.handled_layers = []
            self.view = task_id
            return

        if len(self.handled_layers) == 0:
            # save a copy of each (parametrized) layer of the model
            for param, mask in zip(self.model.parameters(), self.masks):
                if mask is not None:
                    self.handled_layers.append(copy.deepcopy(param))
                else:
                    self.handled_layers.append(None)

        # apply the masks
        for p, mask in zip(self.model.parameters(), self.masks):
            if mask is None:
                continue
            # set to zero the parameters that are free (have no mask) or whose mask ID is greater than task_id
            p.data *= torch.logical_and(mask <= task_id, mask > 0)

        self.view = task_id
