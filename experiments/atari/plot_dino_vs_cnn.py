import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from argparse import ArgumentParser
import sys

sys.path.append("../../")
from utils import plt_style


def parse_args():
    # fmt: off
    parser = ArgumentParser()
    parser.add_argument("--dirname", type=str, default="data/dino_vs_cnn",
        help="directory where the CSV data of the experiments is stored")
    # fmt: on
    return parser.parse_args()


def ma(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def smooth(x, y, num_points, ma_num=10, idxs=None):
    # apply moving average smoothing
    y = ma(y, ma_num)
    x = x[0 : len(y)]

    # only select some points
    idx = (
        np.linspace(0, len(y) - 1, num=num_points, dtype=np.int64)
        if idxs is None
        else idxs
    )
    y = y[idx]
    x = x[idx]
    return x, y


def select_valid(df, data_lbl):
    val_idx = ~pd.isna(df[data_lbl])
    y = np.array(df[data_lbl][val_idx])
    x = np.array(df["global_step"][val_idx])
    return x, y


def plot_ep_rets(fname):
    df = pd.read_csv(fname)
    cnn_mean_cols = []
    dino_mean_cols = []
    for col in df.columns:
        if col.endswith("(DINO-3 layers) - charts/episodic_return"):
            dino_mean_cols.append(col)
        elif col.endswith("(CNN) - charts/episodic_return"):
            cnn_mean_cols.append(col)

    plt.rcParams.update({"font.size": 13})
    fig = plt.figure()
    plt_style.style(fig, legend=False)

    for alg, alg_cols in [("CNN", cnn_mean_cols), ("DINO", dino_mean_cols)]:
        for col in alg_cols:
            x, y = select_valid(df, col)
            x, y = smooth(x, y, num_points=100, ma_num=100)
            plt.plot(x, y, c="tab:blue" if alg == "CNN" else "tab:orange", alpha=0.7)

    custom_lines = [
        Line2D([0], [0], color="tab:blue"),
        Line2D([0], [0], color="tab:orange"),
    ]
    plt.legend(
        custom_lines, ["CNN", "DINO"], fancybox=False, frameon=False, loc="lower right"
    )
    plt.xlabel("Timestep")
    plt.ylabel("Episodic return")
    plt.savefig("dino_vs_cnn.pdf")
    plt.show()


def plot_times(fname, col_names="Name", col_times="Relative Time (Process)"):
    df = pd.read_csv(fname)

    t_cnn = []
    t_dino = []
    for name in df[col_names]:
        t = float(df[df[col_names] == name][col_times].iloc[0])
        t /= 60 * 60
        if "CNN" in name:
            t_cnn.append(t)
        elif "DINO" in name:
            t_dino.append(t)

    plt.rcParams.update({"font.size": 13})
    fig = plt.figure(figsize=(4, 5))
    plt_style.style(fig, legend=False)

    sns.barplot(pd.DataFrame({"CNN": t_cnn, "DINO": t_dino}))
    plt.ylabel("Elapsed time (hours)")
    plt.savefig("dino_vs_cnn_times.pdf")
    plt.show()


if __name__ == "__main__":
    args = parse_args()

    plot_ep_rets(fname=f"{args.dirname}/ep_rets.csv")
    plot_times(fname=f"{args.dirname}/times.csv")
