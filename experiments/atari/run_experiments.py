import subprocess
import argparse
import random
from task_utils import TASKS


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm", type=str, choices=["componet", "finetune", "from-scratch", "prog-net", "packnet"], required=True)
    parser.add_argument("--env", type=str, choices=["ALE/SpaceInvaders-v5", "ALE/Freeway-v5"], default="ALE/SpaceInvaders-v5")
    parser.add_argument("--seed", type=int, required=False, default=None)

    parser.add_argument("--start-mode", type=int, required=True)
    parser.add_argument("--first-mode", type=int, required=True)
    parser.add_argument("--last-mode", type=int, required=True)
    # fmt: on
    return parser.parse_args()


args = parse_args()

modes = TASKS[args.env]
start_mode = args.start_mode

if args.algorithm == "finetune":
    model_type = "cnn-simple-ft"
elif args.algorithm == "componet":
    model_type = "cnn-componet"
elif args.algorithm == "from-scratch":
    model_type = "cnn-simple"
elif args.algorithm == "prog-net":
    model_type = "prog-net"
elif args.algorithm == "packnet":
    model_type = "packnet"

seed = random.randint(0, 1e6) if args.seed is None else args.seed

run_name = (
    lambda task_id: f"{args.env.replace('/', '-')}_{task_id}__{model_type}__run_ppo__{seed}"
)
timesteps = int(1e6)

first_idx = modes.index(start_mode)
for i, task_id in enumerate(modes[first_idx:]):
    params = f"--track --model-type={model_type} --env-id={args.env} --seed={seed}"
    params += f" --mode={task_id} --save-dir=agents --total-timesteps={timesteps}"

    # algorithm specific CLI arguments
    if args.algorithm == "componet":
        params += " --componet-finetune-encoder"
    if args.algorithm == "packnet":
        params += f" --total-task-num={len(modes)}"

    if first_idx > 0 or i > 0:
        # multiple previous modules
        if args.algorithm in ["componet", "prog-net"]:
            params += " --prev-units"
            for i in modes[: modes.index(task_id)]:
                params += f" agents/{run_name(i)}"
        # single previous module
        elif args.algorithm in ["finetune", "packnet"]:
            params += f" --prev-units agents/{run_name(task_id-1)}"

    # Launch experiment
    cmd = f"python3 run_ppo.py {params}"
    print(cmd)
    res = subprocess.run(cmd.split(" "))
    if res.returncode != 0:
        print(f"*** Process returned code {res.returncode}. Stopping on error.")
        quit(1)
