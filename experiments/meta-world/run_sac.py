# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import pathlib
from torch.utils.tensorboard import SummaryWriter
from typing import Literal, Optional, Tuple
from models import shared, SimpleAgent, CompoNetAgent, PackNetAgent, ProgressiveNetAgent
from tasks import get_task, get_task_name
from stable_baselines3.common.buffers import ReplayBuffer


@dataclass
class Args:
    model_type: Literal["simple", "finetune", "componet", "packnet", "prognet"]
    """The name of the NN model to use for the agent"""
    save_dir: Optional[str] = None
    """If provided, the model will be saved in the given directory"""
    prev_units: Tuple[pathlib.Path, ...] = ()
    """Paths to the previous models. Not required when model_type is `simple` or `packnet` or `prognet`"""

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cw-sac"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    task_id: int = 0
    """ID number of the task"""
    eval_every: int = 10_000
    """Evaluate the agent in determinstic mode every X timesteps"""
    num_evals: int = 10
    """Number of times to evaluate the agent"""
    total_timesteps: int = int(1e6)
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5_000
    """timestep to start learning"""
    random_actions_end: int = 10_000
    """timesteps to take actions randomly"""
    policy_lr: float = 1e-3
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


def make_env(task_id):
    def thunk():
        env = get_task(task_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.fc = shared(
            np.array(envs.observation_space.shape).prod()
            + np.prod(envs.action_space.shape)
        )
        self.fc_out = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.fc(x)
        x = self.fc_out(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(nn.Module):
    def __init__(self, envs, model):
        super().__init__()
        self.model = model

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (envs.single_action_space.high + envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x, **kwargs):
        mean, log_std = self.model(x, **kwargs)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, **kwargs):
        mean, log_std = self(x, **kwargs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


@torch.no_grad()
def eval_agent(agent, test_env, num_evals, global_step, writer, device):
    obs, _ = test_env.reset()
    avg_ep_ret = 0
    avg_success = 0
    ep_ret = 0
    for _ in range(num_evals):
        while True:
            obs = torch.Tensor(obs).to(device).unsqueeze(0)
            action, _ = agent(obs)
            obs, reward, termination, truncation, info = test_env.step(
                action[0].cpu().numpy()
            )

            ep_ret += reward

            if termination or truncation:
                avg_success += info["success"]
                avg_ep_ret += ep_ret
                # resets
                obs, _ = test_env.reset()
                ep_ret = 0
                break
    avg_ep_ret /= num_evals
    avg_success /= num_evals
    print(f"\nTEST: ep_ret={avg_ep_ret}, success={avg_success}\n")
    writer.add_scalar("charts/test_episodic_return", avg_ep_ret, global_step)
    writer.add_scalar("charts/test_success", avg_success, global_step)


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"task_{args.task_id}__{args.model_type}__{args.exp_name}__{args.seed}"
    print(f"\n*** Run name: {run_name} ***\n")
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"*** Device: {device}")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.task_id)])
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    # select the model to use as the agent
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    act_dim = np.prod(envs.single_action_space.shape)
    print(f"*** Loading model `{args.model_type}` ***")
    if args.model_type in ["finetune", "componet"]:
        assert (
            len(args.prev_units) > 0
        ), f"Model type {args.model_type} requires at least one previous unit"

    if args.model_type == "simple":
        model = SimpleAgent(obs_dim=obs_dim, act_dim=act_dim).to(device)

    elif args.model_type == "finetune":
        model = SimpleAgent.load(
            args.prev_units[0], map_location=device, reset_heads=True
        ).to(device)

    elif args.model_type == "componet":
        model = CompoNetAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            prev_paths=args.prev_units,
            map_location=device,
        ).to(device)
    elif args.model_type == "packnet":
        packnet_retrain_start = args.total_timesteps - int(args.total_timesteps * 0.2)
        if len(args.prev_units) == 0:
            model = PackNetAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                task_id=args.task_id,
                total_task_num=20,
                device=device,
            ).to(device)
        else:
            model = PackNetAgent.load(
                args.prev_units[0],
                task_id=args.task_id + 1,
                restart_heads=True,
                freeze_bias=True,
                map_location=device,
            ).to(device)
    elif args.model_type == "prognet":
        model = ProgressiveNetAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            prev_paths=args.prev_units,
            map_location=device,
        ).to(device)

    actor = Actor(envs, model).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.random_actions_end:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            if args.model_type == "componet" and global_step % 1000 == 0:
                actions, _, _ = actor.get_action(
                    torch.Tensor(obs).to(device),
                    writer=writer,
                    global_step=global_step,
                )
            else:
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}, success={info['success']}"
                )
                writer.add_scalar(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", info["episode"]["l"], global_step
                )
                writer.add_scalar("charts/success", info["success"], global_step)

                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data.next_observations
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - alpha * next_state_log_pi
                )
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    if args.model_type == "packnet":
                        if global_step >= packnet_retrain_start:
                            # can be called multiple times, only the first counts
                            actor.model.start_retraining()
                        actor.model.before_update()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (
                            -log_alpha.exp() * (log_pi + target_entropy)
                        ).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 100 == 0:
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    "losses/qf2_values", qf2_a_values.mean().item(), global_step
                )
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )

    [
        eval_agent(actor, envs.envs[i], args.num_evals, global_step, writer, device)
        for i in range(envs.num_envs)
    ]

    envs.close()
    writer.close()

    if args.save_dir is not None:
        print(f"Saving trained agent in `{args.save_dir}` with name `{run_name}`")
        actor.model.save(dirname=f"{args.save_dir}/{run_name}")
