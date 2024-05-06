from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib
from tqdm import tqdm
import numpy as np
import os
from argparse import ArgumentParser
from process_results import (
    parse_metadata,
    parse_tensorboard,
    smooth_avg,
    remove_nan,
    areas_up_down,
)


def parse_args():
    # fmt: off
    parser = ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="runs",
        help="directory where the TensorBoard logs are stored")
    parser.add_argument("--save-csv", type=str, default="data/data_transfer_matrix_raw.csv",
        help="directory where the TensorBoard logs are stored")
    parser.add_argument("--smoothing-window", type=int, default=100,
        help="smoothing window for the success rate curves. \
        Defaults to the value used in processing meta-world runs.")
    # fmt: on
    return parser.parse_args()


def reference_transfer(F):
    n = F.shape[0]
    s = []
    for i in range(1, n):
        r = []
        for j in range(i):
            r.append(F[j, i])
        s.append(max(r))
    return np.sum(s) / n


if __name__ == "__main__":
    scalar = "charts/success"

    args = parse_args()

    if not os.path.exists(args.save_csv):
        dfs = []
        for path in tqdm(list(pathlib.Path(args.runs_dir).rglob("*events.out*"))):
            # print(path)

            res = parse_tensorboard(str(path), [scalar])
            if res is not None:
                dic, md = res
            else:
                print("No data. Skipping...")
                continue

            first_task = md["task_id"]
            seed = md["seed"]
            model_type = md["model_type"]

            if model_type == "finetune":
                second_task = int(md["prev_units"][12:-4].split("_")[1])
            else:
                second_task = None

            df = dic[scalar]
            df["first task"] = first_task
            df["second task"] = second_task
            df["seed"] = seed
            df["model"] = model_type
            dfs.append(df)

        df = pd.concat(dfs)
        df.to_csv(args.save_csv, index=False)
    else:
        print(f"Using cache CSV at: {args.save_csv}")
        df = pd.read_csv(args.save_csv)

    #
    # Compute forward transfers
    #

    F = np.zeros((10, 10))
    for first_task in range(10):
        for second_task in range(10):
            baseline = df[(df["model"] == "simple") & (df["first task"] == second_task)]

            method = df[
                (df["model"] == "finetune")
                & (df["first task"] == first_task)
                & (df["second task"] == second_task)
            ]

            # get the curve of the `simple` method
            x_baseline, y_baseline, _ = smooth_avg(
                baseline, xkey="step", ykey="value", w=args.smoothing_window
            )
            x_baseline, y_baseline = remove_nan(x_baseline, y_baseline)

            baseline_area_down = np.trapz(y=y_baseline, x=x_baseline)
            baseline_area_up = np.max(x_baseline) - baseline_area_down

            x_method, y_method, _ = smooth_avg(
                method, xkey="step", ykey="value", w=args.smoothing_window
            )
            x_method, y_method = remove_nan(x_method, y_method)

            # this can happen if a method hasn't the results for all tasks
            if len(x_baseline) > len(x_method):
                print(f"Skipping first_task={first_task}, second_task={second_task}")
                continue

            area_up, area_down = areas_up_down(
                x_method,
                y_method,
                x_baseline,
                y_baseline,
            )

            ft = (area_up - area_down) / baseline_area_up

            F[first_task, second_task] = ft

    rt = reference_transfer(F)
    print("\n=> Reference transfer:", rt)
    print()

    fs = 14
    plt.rc("axes", labelsize=fs)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=fs)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=fs)  # fontsize of the tick labels

    F = np.round(F, 2)
    sns.heatmap(F, annot=True, cmap="RdYlGn", center=0.0, vmin=-1, vmax=1)
    plt.xlabel("Second task")
    plt.ylabel("First task")
    plt.savefig(f"transfer_matrix_metaworld.pdf", pad_inches=0, bbox_inches="tight")
    plt.show()
