from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib
from tqdm import tqdm
import numpy as np
import sys, os
from argparse import ArgumentParser


def parse_args():
    # fmt: off
    parser = ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="runs",
        help="directory where the TensorBoard logs are stored")
    parser.add_argument("--save-csv", type=str, default="data/data_transfer_matrix_raw.csv",
        help="directory where the TensorBoard logs are stored")
    # fmt: on
    return parser.parse_args()


def parse_metadata(ea):
    md = ea.Tensors("hyperparameters/text_summary")[0]
    md_bytes = md.tensor_proto.SerializeToString()

    # remove first non-ascii characters and parse
    start = md_bytes.index(b"|")
    md_str = md_bytes[start:].decode("ascii")

    md = {}
    for row in md_str.split("\n")[2:]:
        s = row.split("|")[1:-1]
        k = s[0]
        if s[1].isdigit():
            v = int(s[1])
        elif s[1].replace(".", "").isdigit():
            v = float(s[1])
        elif s[1] == "True" or s[1] == "False":
            v = s[1] == "True"
        else:
            v = s[1]
        md[k] = v
    return md


def parse_tensorboard(path, scalars, single_pts=[]):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()

    # make sure the scalars are in the event accumulator tags
    if sum([s not in ea.Tags()["scalars"] for s in scalars]) > 0:
        print(f"** Scalar not found. Skipping file {path}")
        return None

    md = parse_metadata(ea)

    for name in single_pts:
        if name in ea.Tags()["scalars"]:
            md[name] = pd.DataFrame(ea.Scalars(name))["value"][0]

    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}, md


def areas_up_down(method_x, method_y, baseline_x, baseline_y):
    up_idx = method_y > baseline_y
    down_idx = method_y < baseline_y

    assert (
        method_x == baseline_x
    ).all(), "The X axis of the baseline and method must be equal."
    x = method_x

    area_up = np.trapz(y=method_y[up_idx], x=x[up_idx]) - np.trapz(
        y=baseline_y[up_idx], x=x[up_idx]
    )
    area_down = np.trapz(y=baseline_y[down_idx], x=x[down_idx]) - np.trapz(
        y=method_y[down_idx], x=x[down_idx]
    )
    return area_up, area_down


def remove_nan(x, y):
    no_nan = ~np.isnan(y)
    return x[no_nan], y[no_nan]


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def chunk_average(x, w):
    split = np.array_split(x, x.shape[0] // w)
    x_avg = np.array([chunk.mean() for chunk in split])
    x_std = np.array([chunk.std() for chunk in split])
    return x_avg, x_std


def compute_success(df, xkey, ykey, score, ma_w, chunk_w, ma_w_extra):
    g = df.groupby(xkey)[ykey]
    mean = g.mean()
    y = mean.values
    x = mean.reset_index()[xkey].values

    x = moving_average(x, w=ma_w)
    y = moving_average(y, w=ma_w)

    y = y >= score

    y, _ = chunk_average(y, chunk_w)
    x, _ = chunk_average(x, chunk_w)

    x = moving_average(x, w=ma_w)
    y = moving_average(y, w=ma_w)

    x = np.insert(x, 0, 0.0)
    y = np.insert(y, 0, 0.0)

    return remove_nan(x, y)


def transfer_matrix(df, success_scores, ma_w=10, chunk_w=30, ma_w_extra=30):
    num_tasks = df["first task"].max() + 1

    F = np.zeros((num_tasks, num_tasks))
    for first_task in range(num_tasks):
        for second_task in range(num_tasks):
            # get the success score for this task
            score = float(
                success_scores[success_scores["task"] == second_task][
                    "success score"
                ].iloc[0]
            )

            # baseline method trained in the second task
            baseline = df[
                (df["model"] == "cnn-simple") & (df["first task"] == second_task)
            ]
            x_baseline, y_baseline = compute_success(
                baseline,
                xkey="step",
                ykey="value",
                score=score,
                ma_w=ma_w,
                chunk_w=chunk_w,
                ma_w_extra=ma_w_extra,
            )

            baseline_area_down = np.trapz(y=y_baseline, x=x_baseline)
            baseline_area_up = np.max(x_baseline) - baseline_area_down

            # get the method's data
            method = df[
                (df["model"] == "cnn-simple-ft")
                & (df["first task"] == first_task)
                & (df["second task"] == second_task)
            ]

            if len(method["seed"].unique()) != len(baseline["seed"].unique()):
                print(f"* Skipping, first={first_task}, second={second_task}")
                continue

            # episodic return to success
            x_method, y_method = compute_success(
                method,
                xkey="step",
                ykey="value",
                score=score,
                ma_w=ma_w,
                chunk_w=chunk_w,
                ma_w_extra=ma_w_extra,
            )

            #
            # Compute forward transfer
            #

            # get a common X axis
            x = []
            mi, bi = 0, 0
            while mi < len(x_method) and bi < len(x_baseline):
                if x_method[mi] < x_baseline[bi]:
                    x.append(x_method[mi])
                    mi += 1
                else:
                    x.append(x_baseline[bi])
                    bi += 1
            x = np.array(x)
            y_b = np.interp(x, x_baseline, y_baseline)
            y_m = np.interp(x, x_method, y_method)

            # areas where the method's curve is above or below the baseline
            up_idx = y_m > y_b
            down_idx = y_m < y_b

            # compute the integrals
            area_up = np.trapz(y=y_m[up_idx], x=x[up_idx]) - np.trapz(
                y=y_b[up_idx], x=x[up_idx]
            )
            area_down = np.trapz(y=y_b[down_idx], x=x[down_idx]) - np.trapz(
                y=y_m[down_idx], x=x[down_idx]
            )
            ft = (area_up - area_down) / baseline_area_up

            F[first_task, second_task] = ft
    return F


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
    args = parse_args()

    scalar = "charts/episodic_return"

    #
    # Convert tensorboard's data to a useful CSV
    #
    if not os.path.exists(args.save_csv):
        dfs = []
        for path in tqdm(list(pathlib.Path(args.runs_dir).rglob("*events.out*"))):
            res = parse_tensorboard(str(path), [scalar])
            if res is not None:
                dic, md = res
            else:
                print("No data. Skipping...")
                continue

            first_task = md["mode"]
            seed = md["seed"]

            model_type = md["model_type"]
            env = md["env_id"]

            if model_type == "cnn-simple-ft":
                second_task = int(md["prev_units"][12:-4].split("_")[1])
            else:
                second_task = None

            # print(seed, first_task, second_task)

            df = dic[scalar]
            df["first task"] = first_task
            df["second task"] = second_task
            df["seed"] = seed
            df["model"] = model_type
            df["env"] = env
            dfs.append(df)

        df = pd.concat(dfs)
        df.to_csv(args.save_csv, index=False)
    else:
        print(f"** Using the cache file in {args.save_csv}")
        df = pd.read_csv(args.save_csv)

    # default values used in these envs (see process_results.py)
    ma_w = 10
    chunk_w = 30
    ma_w_extra = 30

    fs = 14
    plt.rc("axes", labelsize=fs)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=fs)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=fs)  # fontsize of the tick labels

    for env in ["Freeway", "SpaceInvaders"]:
        success_scores = pd.read_csv(f"data/success_scores_{env}.csv")
        data = df[df["env"] == f"ALE/{env}-v5"]

        F = transfer_matrix(data, success_scores, ma_w, chunk_w, ma_w_extra)

        rt = reference_transfer(F)
        print("\n=> Reference transfer:", rt)
        print()
        F = np.round(F, 2)
        sns.heatmap(F, annot=True, cmap="RdYlGn", center=0.0, vmin=-1, vmax=1)
        plt.xlabel("Second task")
        plt.ylabel("First task")
        plt.savefig(f"transfer_matrix_{env}.pdf", pad_inches=0, bbox_inches="tight")
        plt.show()
