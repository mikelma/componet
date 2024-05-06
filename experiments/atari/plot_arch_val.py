import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import sys, os

sys.path.append("../../")
from utils import plt_style


def parse_args():
    # fmt: off
    parser = ArgumentParser()
    parser.add_argument("--dir-prefix", type=str, default="data/arch-val",
        help="directory where the CSV data of the architectural validation experiments is stored")
    # fmt: on
    return parser.parse_args()


def smooth_curve(df, data_col, idx_col="global_step", chunksize=None, w=None):
    idx = ~df[data_col].isnull()
    y = df[data_col][idx].values
    x = df[idx_col][idx].values

    # compute the average of chunks
    if chunksize is not None:
        n = x.shape[0]
        x = np.array([c.mean() for c in np.array_split(x, n // chunksize)])
        y = np.array([c.mean() for c in np.array_split(y, n // chunksize)])

    # second smotthing with a moving average
    if w is not None:
        x = np.convolve(x, np.ones(w), "valid") / w
        y = np.convolve(y, np.ones(w), "valid") / w
    return x, y


def plot_ep_ret(fig, ax, path="data/ep_ret.csv", save_prefix="", legend_loc="best", figlabel="(i)"):
    df = pd.read_csv(path)
    cols = df.columns
    col_method = cols[
        cols.str.contains("cnn-componet") & cols.str.endswith("episodic_return")
    ][0]
    col_ref = cols[
        cols.str.contains("cnn-simple") & cols.str.endswith("episodic_return")
    ][0]

    x, y_ref = smooth_curve(df, data_col=col_ref, chunksize=100, w=10)
    x_min, y_min = smooth_curve(df, data_col=col_ref + "__MIN", chunksize=100, w=10)
    x_max, y_max = smooth_curve(df, data_col=col_ref + "__MAX", chunksize=100, w=10)
    ax.plot(x, y_ref, label="Baseline")
    ax.fill_between(x, y_min, y_max, alpha=0.2)

    x, y_method = smooth_curve(df, data_col=col_method, chunksize=100, w=10)
    x_min, y_min = smooth_curve(df, data_col=col_method + "__MIN", chunksize=100, w=10)
    x_max, y_max = smooth_curve(df, data_col=col_method + "__MAX", chunksize=100, w=10)
    ax.plot(
        x,
        y_method,
        label="CompoNet",
        marker="X",
        markersize=10,
        markeredgecolor="white",
        markevery=len(x) // 5,
    )
    ax.fill_between(x, y_min, y_max, alpha=0.2)

    ax.set_xlabel(f"Timestep\n{figlabel}")
    ax.set_ylabel("Episodic return")

    # place a text box in upper left in axes coords
    # props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    # ax.text(
    #     0.02,
    #     0.97,
    #     figlabel,
    #     transform=ax.transAxes,
    #     fontsize=14,
    #     verticalalignment="top",
    #     bbox=props,
    # )

    plt_style.style(fig, ax, legend=False)
    ax.legend(fancybox=False, frameon=False, loc=legend_loc)


def plot_matches(fig, ax, path="data/matches.csv", save_prefix="", figlabel="(ii)"):
    df = pd.read_csv(path)
    cols = df.columns
    col_out_m_head_out = cols[cols.str.endswith("out_matches_head_out")][0]
    col_out_m_int_pol = cols[cols.str.endswith("out_matches_int_pol")][0]
    col_int_pol_m_head = cols[cols.str.endswith("int_pol_matches_head")][0]
    cols = [col_out_m_head_out, col_out_m_int_pol, col_int_pol_m_head]
    # labels = ["out = out-head", "out = int-pol", "out-head = int-pol"]
    labels = ["Out = Out head", "Out = Int. pol.", "Out head = Int. pol."]
    markers = [None, "D", "o"]

    for i, (col, lbl) in enumerate(zip(cols, labels)):
        x, y = smooth_curve(df, data_col=col, chunksize=10, w=None)

        x_min, y_min = smooth_curve(df, data_col=col + "__MIN", chunksize=10, w=3)
        _, y_max = smooth_curve(df, data_col=col + "__MAX", chunksize=10, w=3)

        select = x <= (1e6 / 3)
        std_select = x_min <= (1e6 / 3)

        ax.plot(
            x[select],
            y[select],
            label=lbl,
            marker=markers[i],
            markersize=8,
            markeredgecolor="white",
            markevery=len(x[select]) // 5,
        )
        ax.fill_between(
            x_min[std_select], y_min[std_select], y_max[std_select], alpha=0.2
        )

    ax.set_xlabel(f"Timestep\n{figlabel}")
    ax.set_ylabel("Matching rate")

    plt_style.style(fig, ax, legend=False, force_sci_x=True)
    ax.legend(fancybox=False, frameon=False)

    # props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    # ax.text(
    #     0.02,
    #     0.97,
    #     figlabel,
    #     transform=ax.transAxes,
    #     fontsize=14,
    #     verticalalignment="top",
    #     bbox=props,
    # )


def plot_attention(
    fif,
    ax,
    path=None,
    is_input=True,
    save_prefix="",
    mark_last=True,
    legend_cols=1,
    legend_loc=None,
    figlabel="(iii)"
):
    if path is None:
        path = "data/input_attentions.csv" if is_input else "data/output_attentions.csv"
    df = pd.read_csv(path)

    cols = df.columns
    num_atts = len(cols[cols.str.contains("att_") & ~cols.str.contains("__")])

    for i in range(num_atts):
        col = cols[cols.str.endswith(f"att_{'in' if is_input else 'out'}_{i}")][0]
        x, y = smooth_curve(df, data_col=col, chunksize=10, w=None)
        x_std, y_min = smooth_curve(df, data_col=col + "__MIN", chunksize=20, w=10)
        _, y_max = smooth_curve(df, data_col=col + "__MAX", chunksize=20, w=10)
        if is_input:
            lbl = f"Prev. {i}" if i < num_atts - 1 else "Out head"
        else:
            lbl = f"Prev. {i}"

        if i == 4 and mark_last:
            kwargs = dict(
                marker="^",
                markersize=10,
                markeredgecolor="white",
                markevery=len(x) // 5,
            )
            lbl = "Inf. Mod."
        elif lbl == "Out head":
            kwargs = dict(
                marker="o",
                markersize=8,
                markeredgecolor="white",
                markevery=len(x) // 5,
            )
        else:
            kwargs = dict()

        ax.plot(x, y, label=lbl, **kwargs)
        ax.fill_between(x_std, y_min, y_max, alpha=0.2)

    ax.set_ylim(0, 1.1)
    ax.set_xlabel(f"Timestep\n{figlabel}")

    l = "In." if is_input else "Out"
    ax.set_ylabel(f"{l} head's attention val.")

    plt_style.style(fig, ax, legend=False)
    if legend_loc is None:
        legend_loc = "upper right" if is_input else "best"
    ax.legend(fancybox=False, frameon=False, loc=legend_loc, ncols=legend_cols)

    # props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    # ax.text(
    #     0.05,
    #     0.95,
    #     figlabel,
    #     transform=ax.transAxes,
    #     fontsize=14,
    #     verticalalignment="top",
    #     bbox=props,
    # )


if __name__ == "__main__":
    args = parse_args()
    for i, dirname in enumerate(
        [
            f"{args.dir_prefix}/data_all_prevs_to_noise",
            f"{args.dir_prefix}/data_noise_except_task4",
        ]
    ):
        plt.rcParams.update({"font.size": 12, "axes.labelsize": 13.5})

        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 6))

        plot_ep_ret(
            fig,
            axs[0, 0],
            path=f"{dirname}/ep_ret.csv",
            save_prefix=dirname + "_",
            legend_loc="best" if i == 1 else "lower right",
            figlabel="(a)" if i == 1 else "(e)"
        )
        plot_matches(
            fig, axs[0, 1],
            path=f"{dirname}/matches.csv",
            save_prefix=dirname + "_",
            figlabel="(b)" if i == 1 else "(f)"
        )
        plot_attention(
            fig,
            axs[1, 0],
            path=f"{dirname}/input_attentions.csv",
            is_input=True,
            save_prefix=dirname + "_",
            mark_last=i == 1,
            legend_cols=2 if i == 1 else 1,
            legend_loc="center right" if i == 1 else "best",
            figlabel="(c)" if i == 1 else "(g)",
        )
        plot_attention(
            fig,
            axs[1, 1],
            path=f"{dirname}/output_attentions.csv",
            is_input=False,
            save_prefix=dirname + "_",
            mark_last=i == 1,
            figlabel="(d)" if i == 1 else "(h)"
        )

        plt.tight_layout()

        plt.savefig(f"{os.path.basename(dirname)}.pdf", pad_inches=0, bbox_inches="tight")
        plt.show()
