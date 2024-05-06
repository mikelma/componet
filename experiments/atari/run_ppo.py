# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from typing import Literal, Tuple, Optional
import pathlib

from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from models import (
    CnnSimpleAgent,
    DinoSimpleAgent,
    CnnCompoNetAgent,
    ProgressiveNetAgent,
    PackNetAgent,
)


@dataclass
class Args:
    # Model type
    model_type: Literal[
        "cnn-simple",
        "cnn-simple-ft",
        "dino-simple",
        "cnn-componet",
        "prog-net",
        "packnet",
    ]
    """The name of the model to use as agent."""
    dino_size: Literal["s", "b", "l", "g"] = "s"
    """Size of the dino model (only needed when using dino)"""
    save_dir: str = None
    """Directory where the trained model will be saved. If not provided, the model won't be saved"""
    prev_units: Tuple[pathlib.Path, ...] = ()
    """Paths to the previous models. Only used when employing a CompoNet or cnn-simple-ft (finetune) agent"""
    mode: int = None
    """Playing mode for the Atari game. The default mode is used if not provided"""
    componet_finetune_encoder: bool = False
    """Whether to train the CompoNet's encoder from scratch of finetune it from the encoder of the previous task"""
    total_task_num: Optional[int] = None
    """Total number of tasks, required when using PackNet"""
    prevs_to_noise: Optional[int] = 0
    """Number of previous policies to set to randomly selected distributions, only valid when model_type is `cnn-componet`"""

    # Experiment arguments
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
    wandb_project_name: str = "ppo-atari"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = int(1e6)
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, mode=None, dino=False):
    def thunk():
        if mode is None:
            env = gym.make(env_id)
        else:
            env = gym.make(env_id, mode=mode)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)

        if not dino:
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
        else:
            env = gym.wrappers.ResizeObservation(env, (224, 224))
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    m = f"_{args.mode}" if args.mode is not None else ""
    run_name = f"{args.env_id.replace('/', '-')}{m}__{args.model_type}__{args.exp_name}__{args.seed}"
    print("*** Run's name:", run_name)
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

    # env setup
    dino = "dino" in args.model_type
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id, i, args.capture_video, run_name, mode=args.mode, dino=dino
            )
            for i in range(args.num_envs)
        ],
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    print(f"*** Model: {args.model_type} ***")
    if args.model_type == "cnn-simple":
        agent = CnnSimpleAgent(envs).to(device)
    elif args.model_type == "cnn-simple-ft":
        if len(args.prev_units) > 0:
            agent = CnnSimpleAgent.load(
                args.prev_units[0], envs, load_critic=False, reset_actor=True
            ).to(device)
        else:
            agent = CnnSimpleAgent(envs).to(device)
    elif args.model_type == "dino-simple":
        agent = DinoSimpleAgent(
            envs, dino_size=args.dino_size, frame_stack=4, device=device
        ).to(device)
    elif args.model_type == "cnn-componet":
        agent = CnnCompoNetAgent(
            envs,
            prevs_paths=args.prev_units,
            finetune_encoder=args.componet_finetune_encoder,
            map_location=device,
        ).to(device)
    elif args.model_type == "prog-net":
        agent = ProgressiveNetAgent(
            envs, prevs_paths=args.prev_units, map_location=device
        ).to(device)

    elif args.model_type == "packnet":
        # retraining in 20% of the total timesteps
        packnet_retrain_start = args.total_timesteps - int(args.total_timesteps * 0.2)

        if args.total_task_num is None:
            print("CLI argument `total_task_num` is required when using PackNet.")
            quit(1)

        if len(args.prev_units) == 0:
            agent = PackNetAgent(
                envs,
                task_id=(args.mode + 1),
                is_first_task=True,
                total_task_num=args.total_task_num,
            ).to(device)
        else:
            agent = PackNetAgent.load(
                args.prev_units[0],
                task_id=args.mode + 1,
                restart_actor_critic=True,
                freeze_bias=True,
            ).to(device)
    else:
        print(f"Model type {args.model_type} is not valid.")
        quit(1)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if (
                    args.track
                    and args.model_type == "cnn-componet"
                    and global_step % 100 == 0
                ):
                    action, logprob, _, value = agent.get_action_and_value(
                        next_obs / 255.0,
                        log_writter=writer,
                        global_step=global_step,
                        prevs_to_noise=args.prevs_to_noise,
                    )
                elif args.model_type == "cnn-componet":
                    action, logprob, _, value = agent.get_action_and_value(
                        next_obs / 255.0, prevs_to_noise=args.prevs_to_noise
                    )
                else:
                    action, logprob, _, value = agent.get_action_and_value(
                        next_obs / 255.0
                    )

                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done
            ).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs / 255.0).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if args.model_type == "cnn-componet":
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds] / 255.0,
                        b_actions.long()[mb_inds],
                        prevs_to_noise=args.prevs_to_noise,
                    )
                else:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds] / 255.0, b_actions.long()[mb_inds]
                    )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                if args.model_type == "packnet":
                    if global_step >= packnet_retrain_start:
                        agent.start_retraining()  # can be called multiple times, only the first counts
                    agent.before_update()
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    envs.close()
    writer.close()

    if args.save_dir is not None:
        print(f"Saving trained agent in `{args.save_dir}` with name `{run_name}`")
        agent.save(dirname=f"{args.save_dir}/{run_name}")
