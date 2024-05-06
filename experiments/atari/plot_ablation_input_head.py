import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import sys
from argparse import ArgumentParser

sys.path.append("../../")
from utils import plt_style


def parse_args():
    # fmt: off
    parser = ArgumentParser()
    parser.add_argument("--dirname", type=str, default="data/input_head_ablation",
        help="directory where the CSV data of the experiments is stored")
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


def plot_ep_ret(fig, ax, path="data/ep_ret.csv", path_ref=None):
    df = pd.read_csv(path)
    cols = df.columns
    col_method = cols[
        cols.str.endswith("episodic_return") & cols.str.contains("No ablation")
    ][0]

    col_ablated = cols[
        cols.str.endswith("episodic_return") & cols.str.contains("Ablated")
    ][0]

    if path_ref is None:
        path_ref = path[:-4] + "_main_exp.csv"

    df_ref = pd.read_csv(path_ref)
    cols = df_ref.columns
    col_ref = cols[
        (cols.str.contains(" cnn-simple ")) & (cols.str.endswith("episodic_return"))
    ][0]

    x, y_ref = smooth_curve(df_ref, data_col=col_ref, chunksize=100, w=10)
    x_min, y_min = smooth_curve(df_ref, data_col=col_ref + "__MIN", chunksize=100, w=10)
    x_max, y_max = smooth_curve(df_ref, data_col=col_ref + "__MAX", chunksize=100, w=10)
    ax.plot(x, y_ref, label="Baseline")
    ax.fill_between(x, y_min, y_max, alpha=0.3)

    x, y_method = smooth_curve(df, data_col=col_method, chunksize=100, w=10)
    x_min, y_min = smooth_curve(df, data_col=col_method + "__MIN", chunksize=100, w=10)
    x_max, y_max = smooth_curve(df, data_col=col_method + "__MAX", chunksize=100, w=10)
    ax.plot(
        x,
        y_method,
        label="CompoNet",
        marker="X",
        markevery=len(x) // 5,
        markeredgecolor="white",
        markersize=10,
    )
    ax.fill_between(x, y_min, y_max, alpha=0.3)

    x, y_ref = smooth_curve(df, data_col=col_ablated, chunksize=100, w=10)
    x_min, y_min = smooth_curve(df, data_col=col_ablated + "__MIN", chunksize=100, w=10)
    x_max, y_max = smooth_curve(df, data_col=col_ablated + "__MAX", chunksize=100, w=10)
    ax.plot(
        x,
        y_ref,
        label="Ablated",
        marker="d",
        markevery=len(x) // 5,
        markeredgecolor="white",
        markersize=10,
    )
    ax.fill_between(x, y_min, y_max, alpha=0.3)

    ax.set_xlabel("Timestep\n(i)")
    ax.set_ylabel("Episodic return")

    plt_style.style(fig, ax, legend=False)
    ax.legend(fancybox=False, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    # textstr = "(i)"
    # props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    # ax.text(
    #     0.05,
    #     1.05,
    #     textstr,
    #     transform=ax.transAxes,
    #     fontsize=14,
    #     verticalalignment="top",
    #     bbox=props,
    # )


def plot_matches(fig, ax, path="data/matches.csv"):
    df = pd.read_csv(path)
    variants = ["No ablation", "Ablated"]
    variant_styles = ["solid", "dashed"]
    labels = ["Out = Out head", "Out = Int. pol.", "Out head = Int. pol."]
    markers = [None, "D", "o"]
    num_markers = 5

    c = list(mcolors.TABLEAU_COLORS)
    colors = [c[0], c[1], c[3]]

    for variant, linestyle in zip(variants, variant_styles):
        cols = df.columns[df.columns.str.contains(variant)]
        col_out_m_head_out = cols[cols.str.endswith("out_matches_head_out")][0]
        col_out_m_int_pol = cols[cols.str.endswith("out_matches_int_pol")][0]
        col_int_pol_m_head = cols[cols.str.endswith("int_pol_matches_head")][0]
        cols = [col_out_m_head_out, col_out_m_int_pol, col_int_pol_m_head]

        for i, (col, lbl) in enumerate(zip(cols, labels)):
            x, y = smooth_curve(df, data_col=col, chunksize=5, w=10)
            x_min, y_min = smooth_curve(df, data_col=col + "__MIN", chunksize=5, w=10)
            _, y_max = smooth_curve(df, data_col=col + "__MAX", chunksize=5, w=10)

            color = colors[i % len(labels)]
            ax.plot(
                x,
                y,
                label=lbl,
                linestyle=linestyle,
                color=color,
                marker=markers[i],
                markersize=8,
                markevery=len(x) // num_markers,
                markeredgecolor="white",
            )
            ax.fill_between(x_min, y_min, y_max, alpha=0.1, color=color)

    ax.set_xlabel("Timestep\n(ii)")
    ax.set_ylabel("Matching rate")

    plt_style.style(fig, ax, legend=False)
    # ax.legend(fancybox=False, frameon=False)

    c = colors + ["black"] * 2
    styles = ["solid"] * (len(c) - 1) + ["dashed"]
    custom_lines = [
        Line2D([0], [0], color=c[i], linestyle=styles[i]) for i in range(len(c))
    ]
    custom_lines[1].set(marker=markers[1], markersize=8, markeredgecolor="white")
    custom_lines[2].set(marker=markers[2], markersize=8, markeredgecolor="white")
    ax.legend(
        custom_lines,
        labels + ["Original", "Ablated"],
        fancybox=False,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )
    # textstr = "(ii)"
    # props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    # ax.text(
    #     0.05,
    #     1.05,
    #     textstr,
    #     transform=ax.transAxes,
    #     fontsize=14,
    #     verticalalignment="top",
    #     bbox=props,
    # )


def plot_input_attention(fig, ax, path="data/input_attentions.csv"):
    df = pd.read_csv(path)
    cols = df.columns
    num_atts = 6

    for i in range(num_atts):
        col = cols[cols.str.endswith(f"att_in_{i}")][0]
        x, y = smooth_curve(df, data_col=col, chunksize=10, w=None)
        _, y_min = smooth_curve(df, data_col=col + "__MIN", chunksize=10, w=None)
        _, y_max = smooth_curve(df, data_col=col + "__MAX", chunksize=10, w=None)
        lbl = f"Prev. {i}" if i < num_atts - 1 else "Out head"

        marker = None if i != 4 else "*"
        marker = "o" if lbl == "Out head" else marker
        s = 15 if marker == "*" else 8
        lbl = "Inf. Mod." if i == 4 else lbl

        ax.plot(
            x,
            y,
            label=lbl,
            marker=marker,
            markevery=len(x) // 5,
            markeredgecolor="white",
            markersize=s,
        )
        ax.fill_between(x, y_min, y_max, alpha=0.1)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Timestep\n(iii)")
    ax.set_ylabel(f"In. head's attention val.")

    plt_style.style(fig, ax, legend=False)
    ax.legend(fancybox=False, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    # place a text box in upper left in axes coords
    # textstr = "(iii)"
    # props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    # ax.text(
    #     0.05,
    #     1.05,
    #     textstr,
    #     transform=ax.transAxes,
    #     fontsize=14,
    #     verticalalignment="top",
    #     bbox=props,
    # )


def plot_output_attention(fig, ax, path="data/output_attentions.csv"):
    df = pd.read_csv(path)

    cols = df.columns
    num_atts = 5
    variants = ["No ablation", "Ablated"]
    variant_styles = ["solid", "dashed"]
    labels = [f"Prev. {i}" for i in range(num_atts)]
    colors = list(mcolors.TABLEAU_COLORS)

    for variant, style in zip(variants, variant_styles):
        for i in range(num_atts):
            col = cols[cols.str.endswith(f"att_out_{i}") & cols.str.contains(variant)][
                0
            ]
            x, y = smooth_curve(df, data_col=col, chunksize=10, w=None)
            _, y_min = smooth_curve(df, data_col=col + "__MIN", chunksize=10, w=None)
            _, y_max = smooth_curve(df, data_col=col + "__MAX", chunksize=10, w=None)

            marker = "*" if i == num_atts - 1 else None

            ax.plot(
                x,
                y,
                linestyle=style,
                color=colors[i],
                marker=marker,
                markevery=len(x) // 6,
                markeredgecolor="white",
                markersize=15,
            )
            ax.fill_between(x, y_min, y_max, alpha=0.1, color=colors[i])

    ax.set_ylim(0, 1)
    ax.set_xlabel("Timestep\n(iv)")
    ax.set_ylabel(f"Out head's attention val.")

    plt_style.style(fig, ax, legend=False)
    # ax.legend(loc="center right", fancybox=False, frameon=False)

    c = colors[:num_atts] + ["black"] * 2
    styles = ["solid"] * (num_atts + 1) + ["dashed"]
    custom_lines = [
        Line2D([0], [0], color=c[i], linestyle=styles[i]) for i in range(len(c))
    ]
    custom_lines[-3].set(marker="*", markeredgecolor="white", markersize=15)

    ax.legend(
        custom_lines,
        labels + ["Original", "Ablated"],
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fancybox=False,
        frameon=False,
    )
    # textstr = "(iv)"
    # props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    # ax.text(
    #     0.05,
    #     1.1,
    #     textstr,
    #     transform=ax.transAxes,
    #     fontsize=14,
    #     verticalalignment="top",
    #     bbox=props,
    # )


if __name__ == "__main__":
    args = parse_args()
    plt.rcParams.update({"font.size": 12})

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 6))

    plot_ep_ret(fig, axs[0, 0], path=f"{args.dirname}/ep_ret.csv")
    plot_matches(fig, axs[0, 1], path=f"{args.dirname}/matches.csv")
    plot_input_attention(fig, axs[1, 0], path=f"{args.dirname}/input_head.csv")
    plot_output_attention(fig, axs[1, 1], path=f"{args.dirname}/output_head.csv")

    plt.tight_layout(rect=[0.12, 0, 1.0, 1.0])

    plt.savefig(f"ablation_input_head.pdf")
    plt.show()
