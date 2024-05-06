import gymnasium as gym
import numpy as np
import torch
from models import shared, SimpleAgent, CompoNetAgent, PackNetAgent, ProgressiveNetAgent
from tasks import get_task, get_task_name
import argparse
import os
from run_sac import Actor
import random


def parse_args():
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--load", type=str, required=True)
    parser.add_argument("--task-id", type=int, required=False, default=None)
    parser.add_argument("--seed", type=int, required=False, default=None)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--csv', default=None, type=str)
    # fmt: on

    return parser.parse_args()


def parse_load_name(path):
    name = os.path.basename(path)
    s = name.split("__")
    method = s[1]
    seed = int(s[-1])
    task = int(s[0].split("_")[-1])
    return method, seed, task


def make_env(task_id, render_human=False):
    def thunk():
        env = get_task(task_id)
        if render_human:
            env.render_mode = "human"
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


@torch.no_grad()
def eval_agent(agent, test_env, num_evals, device):
    obs, _ = test_env.reset()
    ep_rets = []
    successes = []
    ep_ret = 0
    for _ in range(num_evals):
        while True:
            obs = torch.Tensor(obs).to(device).unsqueeze(0)
            action, _ = agent(obs)
            action = action[0].cpu().numpy()
            obs, reward, termination, truncation, info = test_env.step(action)

            ep_ret += reward

            if termination or truncation:
                successes.append(info["success"])
                ep_rets.append(ep_ret)
                print(ep_ret, successes[-1])
                # resets
                obs, _ = test_env.reset()
                ep_ret = 0
                break

    print(f"\nTEST: ep_ret={np.mean(ep_rets)}, success={np.mean(successes)}\n")
    return successes


if __name__ == "__main__":
    args = parse_args()

    method, seed, train_task = parse_load_name(args.load)

    task_id = args.task_id if args.task_id is not None else train_task
    seed = args.seed if args.seed is not None else seed

    print(
        f"Method: {method}, seed: {seed}, train task: {train_task}, test task: {task_id}"
    )

    envs = gym.vector.SyncVectorEnv([make_env(task_id, render_human=args.render)])
    env = envs.envs[0]

    # set the seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if method in ["simple", "finetune"]:
        model = SimpleAgent.load(args.load, map_location=device)

    agent = Actor(envs, model)

    successes = eval_agent(agent, env, num_evals=args.num_episodes, device=device)

    if args.csv:
        exists = os.path.exists(args.csv)
        with open(args.csv, "w" if not exists else "a") as f:
            if not exists:
                f.write("algorithm,test task,train task,seed,success\n")
            for v in successes:
                f.write(f"{method},{task_id},{train_task},{seed},{v}\n")
