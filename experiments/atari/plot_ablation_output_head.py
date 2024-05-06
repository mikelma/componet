import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from glob import glob
from argparse import ArgumentParser
import sys
sys.path.append("../../")
from utils import plt_style


def parse_args():
    # fmt: off
    parser = ArgumentParser()
    parser.add_argument("--dirname", type=str, default="data/output_head_ablation",
        help="directory where the CSV data of the experiments is stored")
    # fmt: on
    return parser.parse_args()


def chunk_smoothing(series, chunksize):
    assert chunksize % 2 != 0, "process_data: chunksize must be odd"
    arr = series.values
    chunked = np.array_split(arr, arr.shape[0] // chunksize)
    avgs = [c.mean() for c in chunked]
    stds = [c.std() for c in chunked]
    return np.array(avgs), np.array(stds)


def preprocess_data(df, data_col, idx_col, rw_winsize=100, chunksize=111, last_eps=100):
    non_null = ~df[data_col].isnull()
    df = df[non_null]

    # rolling window smoothing
    df.loc[:, data_col] = df[data_col].rolling(window=rw_winsize).mean()

    # smooth by averaging chunks chunks
    data, data_std = chunk_smoothing(df[data_col], chunksize=chunksize)
    idx, _ = chunk_smoothing(df[idx_col], chunksize=chunksize)

    return pd.DataFrame({
        data_col: data, data_col+"_std": data_std,
        idx_col: idx, # "last avg": last_avg, "last std": last_std
    })


def dir_to_df(dirname, winsize=100, chunk_size=101):
    dfs = []
    for i, path in enumerate(glob(f"{dirname}/*.csv")):
        task_id = int(path.split("/")[-1][5:-4])

        if task_id == 0: continue

        df = pd.read_csv(path)
        data_col = df.columns[
            df.columns.str.contains("cnn-componet")
            & df.columns.str.endswith("episodic_return")
        ][0]
        df = df[[data_col, "global_step"]]
        df = df.rename(columns={data_col: "ep_ret"})

        df = preprocess_data(
            df,
            data_col="ep_ret",
            idx_col="global_step",
            rw_winsize=winsize,
            chunksize=chunk_size,
        )
        df["task_id"] = task_id
        # df["global_step"] += i * total_timesteps
        dfs.append(df)
    return pd.concat(dfs)


def single_file_to_df(path, winsize=100, chunk_size=101):
    df = pd.read_csv(path)
    ep_ret_cols = df.columns[df.columns.str.endswith("episodic_return")]

    dfs = []
    for col in ep_ret_cols:
        task_id = int(col.split(" ")[1])

        sel = df[[col, "global_step"]].copy()
        sel = df.rename(columns={col: "ep_ret"})

        sel = preprocess_data(
            sel,
            data_col="ep_ret",
            idx_col="global_step",
            rw_winsize=winsize,
            chunksize=chunk_size,
        )
        sel["task_id"] = task_id
        dfs.append(sel)
    df = pd.concat(dfs)
    return df

def generate_plot(dir_prefix, env, winsize, chunk_size):
    lbls = ["Original", "Ablated"]
    colors = ["tab:blue", "tab:orange"]

    no_out_head = dir_to_df(f"{dir_prefix}/no_out_head/{env}", chunk_size=chunk_size, winsize=winsize)
    normal = dir_to_df(f"{dir_prefix}/with_out_head/{env}", chunk_size=chunk_size, winsize=winsize)

    dfs = [normal, no_out_head]

    tasks = sorted(normal["task_id"].unique()[1:])

    fig, axes = plt.subplots(nrows=1, ncols=len(tasks), figsize=(20, 3.5))

    for i, task_id in enumerate(tasks):
        ax = axes[i]

        plt_style.style(fig, ax=ax, legend=False)

        if i > 0:
            ax.set_xticks([])
        else:
            ax.set_ylabel("Episodic return", fontsize=13)
            ax.set_xlabel("Timestep", fontsize=13)

        for data in dfs:
            sel = data[data["task_id"] == task_id]
            ax.plot(sel["global_step"], sel["ep_ret"], label="Yes")
            ax.fill_between(
                sel["global_step"],
                sel["ep_ret"] - sel["ep_ret_std"],
                sel["ep_ret"] + sel["ep_ret_std"],
                label="Yes",
                alpha=0.3,
            )

        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=14)
        ax.set_title(f"Task {task_id}")

        # Shrink current axis's height by 10% on the bottom
        p = 0.2  # %
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * p,
                         box.width, box.height * (1-p)])

    custom_lines = [Line2D([0], [0], color=c) for c in colors]
    fig.legend(custom_lines, lbls, fancybox=False, frameon=False,
               loc="outside lower center", ncols=len(lbls), fontsize=14)

    plt.savefig(f"out_head_ablation_curves_{env}.pdf", pad_inches=0, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    generate_plot(args.dirname, "SpaceInvaders", winsize=100, chunk_size=501)
    generate_plot(args.dirname, "Freeway", winsize=50, chunk_size=101)
