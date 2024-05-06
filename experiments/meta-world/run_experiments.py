import subprocess
import argparse
import random
from tasks import tasks


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=[
            "simple",
            "componet",
            "finetune",
            "from-scratch",
            "prognet",
            "packnet",
        ],
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--no-run", default=False, action="store_true")

    parser.add_argument("--start-mode", type=int, required=True)
    return parser.parse_args()


args = parse_args()

modes = list(range(20)) if args.algorithm != "simple" else list(range(10))

# NOTE: If the algoritm is not `simple`, it always should start from the second task
if args.algorithm not in ["simple", "packnet", "prognet"] and args.start_mode == 0:
    start_mode = 1
else:
    start_mode = args.start_mode

run_name = (
    lambda task_id: f"task_{task_id}__{args.algorithm if task_id > 0 or args.algorithm in ['packnet', 'prognet'] else 'simple'}__run_sac__{args.seed}"
)

first_idx = modes.index(start_mode)
for i, task_id in enumerate(modes[first_idx:]):
    params = f"--model-type={args.algorithm} --task-id={task_id} --seed={args.seed}"
    params += f" --save-dir=agents"

    if first_idx > 0 or i > 0:
        # multiple previous modules
        if args.algorithm in ["componet", "prognet"]:
            params += " --prev-units"
            for i in modes[: modes.index(task_id)]:
                params += f" agents/{run_name(i)}"
        # single previous module
        elif args.algorithm in ["finetune", "packnet"]:
            params += f" --prev-units agents/{run_name(task_id-1)}"

    # Launch experiment
    cmd = f"python3 run_sac.py {params}"
    print(cmd)

    if not args.no_run:
        res = subprocess.run(cmd.split(" "))
        if res.returncode != 0:
            print(f"*** Process returned code {res.returncode}. Stopping on error.")
            quit(1)
