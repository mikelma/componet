import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pathlib
import argparse
from tabulate import tabulate
import sys, os

sys.path.append("../../")
from utils import style

SETTINGS = {
    "SpaceInvaders": dict(
        ma_w_1=10,
        num_pts_sc=100,
        sc_percent=1.0,
        chunk_avg_w=30,
        ma_w_extra=30,
        ma_std_extra=10,
    ),
    "Freeway": dict(
        ma_w_1=10,
        num_pts_sc=100,
        sc_percent=1.0,
        chunk_avg_w=30,
        ma_w_extra=10,
        ma_std_extra=None,
    ),
}

METHOD_NAMES = {
    "cnn-simple": "Baseline",
    "cnn-simple-ft": "FT",
    "cnn-componet": "CompoNet",
    "prog-net": "ProgressiveNet",
    "packnet": "PackNet",
}

METHOD_COLORS = {
    "cnn-simple": "darkgray",
    "cnn-simple-ft": "tab:orange",
    "cnn-componet": "tab:blue",
    "prog-net": "tab:green",
    "packnet": "tab:purple",
}

METHOD_ORDER = ["cnn-simple", "cnn-componet", "cnn-simple-ft", "prog-net", "packnet"]


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/envs/Freeway",
        choices=["data/envs/Freeway", "data/envs/SpaceInvaders"],
        help="path to the directory where the CSV of each task is stored")
    parser.add_argument("--eval-results", type=str, default="data/eval_results.csv",
        help="path to the file where the CSV with the evaluation results is located")
    # fmt: on
    return parser.parse_args()


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


def compute_success(
    df,
    ma_w_1=10,
    num_pts_sc=100,
    sc_percent=0.8,
    chunk_avg_w=30,
    ma_w_extra=30,
    ma_std_extra=10,
):
    """Takes the DataFrame of the CSV file form W&B and returns a new
    dataframe with the success of each algorithm in every timestep.

    df           -- DataFrame to process.

    ma_w_1       -- Window size of the moving average applied to the
                    return curves before computing the success score.

    num_pts_sc   -- Number of points to use to compute the success score.

    sc_percent   -- Percentage of the average final return to take as
                    the success score.

    chunk_avg_w  -- Window size of the chunk average smoothing applied to
                    the success curves.

    ma_w_extra   -- Window size of the MA smoothing applied to the success
                    curves.
    ma_std_extra -- Window size of the extra MA applied to the std of
                    the success curves.
    """
    data_cols = df.columns[df.columns.str.endswith("episodic_return")]
    # get the name of the method from the column's name
    methods = [col.split(" ")[1] for col in data_cols]

    # compute the success_score automatically as
    # the `sc_percent` % of the average return of
    # the last 100 episodes of all algorithms in
    # the dataframe
    rets = []
    returns = {}
    for method, col in zip(methods, data_cols):
        x, y = df["global_step"].values, df[col].values
        x, y = remove_nan(x, y)

        x = moving_average(x, w=ma_w_1)
        y = moving_average(y, w=ma_w_1)
        returns[method] = (x, y)

        rets.append(y[:-num_pts_sc].mean())

    success_score = sc_percent * np.mean(rets)

    data = {}
    for method in methods:
        x, y = returns[method]
        y = y >= success_score

        x, _ = chunk_average(x, w=chunk_avg_w)
        y, y_std = chunk_average(y, w=chunk_avg_w)

        if ma_w_extra is not None:
            x = moving_average(x, w=ma_w_extra)
            y = moving_average(y, w=ma_w_extra)
            y_std = moving_average(y_std, w=ma_w_extra)

        if ma_std_extra is not None:
            x_std = moving_average(x, w=10)
            y_min = moving_average(np.maximum(y - y_std, 0), w=10)
            y_max = moving_average(np.minimum(y + y_std, 1), w=10)
        else:
            x_std = x
            y_min = np.maximum(y - y_std, 0)
            y_max = np.minimum(y + y_std, 1)

        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, 0.0)
        y_std = np.insert(y_std, 0, 0.0)
        y_min = np.insert(y_min, 0, 0.0)
        y_max = np.insert(y_max, 0, 0.0)
        x_std = np.insert(x_std, 0, 0.0)

        # plt.plot(x, y, label=method)
        # plt.fill_between(
        #     x_std,
        #     y_min,
        #     y_max,
        #     alpha=0.3
        # )

        d = {}
        d["global_step"] = x
        d["success"] = y
        d["std_high"] = y_max
        d["std_low"] = y_min
        d["std_x"] = x_std
        d["final_success"] = np.mean(y[-100:])
        d["final_success_std"] = np.std(y[-100:])

        data[method] = d

    # plt.gca()
    # plt.legend()
    # plt.show()

    return data, success_score


def compute_forward_transfer(data):
    baseline_method = "cnn-simple"
    methods = list(METHOD_NAMES.keys())

    ft_data = {}
    for task_id in data.keys():
        ft_data[task_id] = {}

        # get the baseline's data
        task_data = data[task_id]
        x_baseline = task_data[baseline_method]["global_step"]
        y_baseline = task_data[baseline_method]["success"]
        baseline_area_down = np.trapz(x=x_baseline, y=y_baseline)
        baseline_area_up = np.max(x_baseline) - baseline_area_down

        for method in methods:
            x_method = task_data[method]["global_step"]
            y_method = task_data[method]["success"]
            y_baseline = task_data[baseline_method]["success"]

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
            y_baseline = np.interp(x, x_baseline, y_baseline)
            y_method = np.interp(x, x_method, y_method)

            # compute the actual FT
            up_idx = y_method > y_baseline
            down_idx = y_method < y_baseline

            area_up = np.trapz(y=y_method[up_idx], x=x[up_idx]) - np.trapz(
                y=y_baseline[up_idx], x=x[up_idx]
            )
            area_down = np.trapz(y=y_baseline[down_idx], x=x[down_idx]) - np.trapz(
                y=y_method[down_idx], x=x[down_idx]
            )
            ft = (area_up - area_down) / baseline_area_up

            ft_data[task_id][method] = ft

    #
    # Printing the results in a pretty table
    #
    methods = list(METHOD_NAMES.keys())
    table = []
    for task_id in sorted(ft_data.keys()):
        row = [task_id]
        for i, method in enumerate(methods):
            val = round(ft_data[task_id][method], 2)
            row.append(val)
        table.append(row)
    table.append([None] * len(method))

    # compute the average and std FT of every method
    avgs = []
    for method in methods:
        method_avg = []
        for task_id in sorted(ft_data.keys())[
            1:
        ]:  # ignore the first task to compute the avg. ft
            method_avg.append(ft_data[task_id][method])
        mean = round(np.mean(method_avg), 2)
        std = round(np.std(method_avg), 2)
        avgs.append(f"{mean} ({std})")
    table.append(["Avg."] + avgs)

    print("\n\n----- FORWARD TRANSFER -----\n")
    print(
        tabulate(
            table,
            headers=["Task ID"] + [METHOD_NAMES[m] for m in methods],
            tablefmt="rounded_outline",
        )
    )

    return ft_data


def compute_final_performance(data):
    methods = list(METHOD_NAMES.keys())
    table = []
    for task_id in sorted(data.keys()):
        row = [task_id]
        for i, method in enumerate(methods):
            val = round(data[task_id][method]["final_success"], 2)
            row.append(val)
        table.append(row)
    table.append([None] * len(method))

    avgs = []
    for j in range(1, len(table[0])):  # skip task id's column
        m = []
        for i in range(len(table) - 1):  # skip Nones row
            m.append(table[i][j])
        mean = round(np.mean(m), 2)
        std = round(np.std(m), 2)
        avgs.append(f"{mean} ({std})")

    table.append(["Avg."] + avgs)

    print("\n\n----- PERFORMANCE -----\n")
    print(
        tabulate(
            table,
            headers=["Task ID"] + [METHOD_NAMES[m] for m in methods],
            tablefmt="rounded_outline",
        )
    )
    print(
        "* NOTE: This is not the final performance in the case of the Finetune method, \nbut the performance at the time of solving each task.\n"
    )


def process_eval(df, data, success_scores, env):
    eval_results = {}
    for method in df["algorithm"].unique():
        print(
            f"\n** Final performance of the \x1b[31;1m{METHOD_NAMES[method]}\x1b[0m method:"
        )

        # per_task = {}
        perf_total = []
        forg_total = []
        perf_by_task = []
        forg_by_task = []
        for task_id in sorted(success_scores.keys()):
            sel = df[
                (df["test mode"] == task_id)
                & (df["environment"] == env)
                & (df["algorithm"] == method)
            ]
            s = sel["ep ret"].values >= success_scores[task_id]

            # the performance when the current task was task_id
            past_perf = data[task_id]["cnn-simple-ft"]["final_success"]

            perf_total += list(s)
            forg_total += list(past_perf - s)

            perf = round(s.mean(), 2)
            perf_std = round(s.std(), 2)
            forg = round((past_perf - s).mean(), 2)
            forg_std = round((past_perf - s).std(), 2)
            print(
                f"  - Task {task_id} => Perf.: {perf} [{perf_std}], Forg.: {forg} [{forg_std}]"
            )

            perf_by_task.append((perf, perf_std))
            forg_by_task.append((forg, forg_std))

        perf = round(np.mean(perf_total), 2)
        perf_std = round(np.std(perf_total), 2)
        forg = round(np.mean(forg_total), 2)
        forg_std = round(np.std(forg_total), 2)
        print(f"  + Avg: Perf.: {perf} [{perf_std}], Forg.: {forg} [{forg_std}]")

        eval_results[method] = {}
        eval_results[method]["perf"] = perf_by_task
        eval_results[method]["forg"] = forg_by_task

    return eval_results


def plot_data(data, save_name="plot.pdf", total_timesteps=1e6):
    methods = METHOD_ORDER
    num_tasks = len(data.keys())
    fig, axes = plt.subplots(nrows=len(methods) + 1, figsize=(10, 8))

    #
    # Plot all the method together
    #
    ax = axes[0]
    for i in range(num_tasks):
        for method in methods:
            offset = i * total_timesteps
            ax.plot(
                data[i][method]["global_step"] + offset,
                data[i][method]["success"],
                c=METHOD_COLORS[method],
                linewidth=0.8,
            )
            ax.set_ylabel("Success")

    ax.set_xticks(
        np.arange(num_tasks) * 1e6,
        [f"{i}" for i in range(num_tasks)],
        fontsize=7,
        color="dimgray",
    )
    ax.vlines(
        x=np.arange(num_tasks) * 1e6,
        ymin=0.0,
        ymax=1,
        colors="tab:gray",
        alpha=0.3,
        linestyles="dashed",
        linewidths=0.7,
    )

    style(fig, ax=ax, legend=False, grid=False, ax_math_ticklabels=False)

    for i, method in enumerate(METHOD_ORDER):
        color = METHOD_COLORS[METHOD_ORDER[i]]
        ax = axes[i + 1]
        ax.vlines(
            x=np.arange(num_tasks) * 1e6,
            ymin=0.0,
            ymax=1,
            colors="tab:gray",
            alpha=0.3,
            linestyles="dashed",
            linewidths=0.7,
        )
        ax.set_xticks(
            np.arange(num_tasks) * 1e6,
            [f"{i}" for i in range(num_tasks)],
            fontsize=7,
            color="dimgray",
        )
        ax.set_ylabel(f"{METHOD_NAMES[method]}\n\nSuccess")

        for task_id in range(num_tasks):
            x = data[task_id][method]["global_step"]
            y = data[task_id][method]["success"]
            y_high = data[task_id][method]["std_high"]
            y_low = data[task_id][method]["std_low"]
            x_std = data[task_id][method]["std_x"]

            offset = task_id * total_timesteps

            ax.plot(x + offset, y, c=color, linewidth=0.8)

            ax.fill_between(
                x_std + offset,
                y_low,
                y_high,
                alpha=0.3,
                color=color,
            )

        style(fig, ax=ax, legend=False, grid=False, ax_math_ticklabels=False)

    # only applied to the last `ax` (plot)
    ax.set_xlabel("Task ID")

    lines = [
        Line2D([0], [0], color=METHOD_COLORS[METHOD_ORDER[i]])
        for i in range(len(methods))
    ]
    fig.legend(
        lines,
        [METHOD_NAMES[m] for m in METHOD_ORDER],
        fancybox=False,
        frameon=False,
        loc="outside lower center",
        ncols=len(methods),
    )

    plt.savefig(save_name, pad_inches=0, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    args = parse_args()

    env = os.path.basename(args.data_dir)

    #
    # Compute the success curve of each method in every task
    #
    data = {}
    scores = {}
    for path in pathlib.Path(args.data_dir).glob("*.csv"):
        task_id = int(str(path)[:-4].split("_")[-1])  # obtain task ID from the path

        df = pd.read_csv(path)

        cfg = dict()
        for k in SETTINGS.keys():
            if k in str(path):
                cfg = SETTINGS[k]
                break

        data_task, success_score = compute_success(df, **cfg)
        data[task_id] = data_task
        scores[task_id] = success_score

    print("\n** Success scores used:")
    [print(round(scores[t], 2), end=" ") for t in sorted(scores.keys())]
    print()
    #
    # Compute forward transfer & final performance
    #
    if os.path.exists(args.eval_results):
        eval_results = process_eval(
            pd.read_csv(args.eval_results), data, scores, f"ALE/{env}-v5"
        )
    else:
        eval_results = None

    ft_data = compute_forward_transfer(data)
    compute_final_performance(data)

    fname = f"summary_data_{env}.csv"
    with open(fname, "w") as f:
        f.write("env,method,task id,perf,perf std,ft,forg,forg std\n")
        for task_id in sorted(list(data.keys())):
            for method in data[0].keys():
                ft = ft_data[task_id][method]
                if eval_results is not None and method in eval_results.keys():
                    perf, perf_std = eval_results[method]["perf"][task_id]
                    forg, forg_std = eval_results[method]["forg"][task_id]
                    if METHOD_NAMES[method] == "Finetune":
                        n_perf = data[task_id][method]["final_success"]
                        n_perf_std = data[task_id][method]["final_success_std"]
                        f.write(
                            f"{env},Finetune-N,{task_id},{n_perf},{n_perf_std},{ft},0,0\n"
                        )
                else:
                    perf = data[task_id][method]["final_success"]
                    perf_std = data[task_id][method]["final_success_std"]
                    forg, forg_std = 0, 0
                f.write(
                    f"{env},{METHOD_NAMES[method]},{task_id},{perf},{perf_std},{ft},{forg},{forg_std}\n"
                )
    print(f"\n*** A summary of the results has been saved to `{fname}` ***\n")

    #
    # Plotting
    #
    env = args.data_dir.split("/")[-1]
    plot_data(data, save_name=f"success_curves_{env}.pdf")
