import torch
import torch.nn as nn
from models.prognet import ProgressiveNet
import sys, os

sys.path.append(os.path.dirname(__file__) + "/../..")
from componet import CompoNet, FirstModuleWrapper
import torch.utils.benchmark as benchmark
import pandas as pd
import argparse


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--act-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument("--test-prevs", type=int, default=[5, 10, 20], nargs="+")
    parser.add_argument("--min-run-time", type=int, default=5)

    parser.add_argument("--save-path", type=str, default="data/benchmarking.csv")

    parser.add_argument("--plot", type=bool, default=False,
        action=argparse.BooleanOptionalAction,
        help="don't run the benchmark and plot the results from `--save-path`")
    parser.add_argument("--joined", type=bool, default=False,
        action=argparse.BooleanOptionalAction,
        help="whether to plot the ")

    # fmt: on

    return parser.parse_args()


def build_prognet(obs_dim, act_dim, hidden_dim, num_prevs, device):
    assert num_prevs > 1

    m1 = ProgressiveNet(input_dim=obs_dim, hidden_dim=hidden_dim, previous_models=[])
    prevs = [m1]

    for _ in range(num_prevs):
        model = ProgressiveNet(
            input_dim=obs_dim, hidden_dim=hidden_dim, previous_models=prevs.copy()
        ).to(device)
        prevs.append(model)

    model = nn.Sequential(model, nn.Linear(hidden_dim, act_dim)).to(device)
    return model


def build_componet(obs_dim, act_dim, hidden_dim, num_prevs, device):
    assert num_prevs > 1
    net = lambda input_dim: nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, act_dim),
    ).to(device)

    m1 = FirstModuleWrapper(net(obs_dim), ret_probs=False).to(device)
    prevs = [m1]
    for _ in range(num_prevs):
        model = CompoNet(
            previous_units=prevs.copy(),
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            out_dim=act_dim,
            internal_policy=net(obs_dim + hidden_dim),
            ret_probs=False,
            proj_bias=True,
        ).to(device)
        prevs.append(model)
    return model


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_params(model):
    return sum(p.numel() for p in model.parameters())


def prognet_total_parameters(model):
    s = sum([total_params(m) for m in model[0].previous_models])
    return s + total_params(model)


def componet_total_parameters(model):
    s = sum([total_params(m) for m in model.previous_units])
    return s + total_params(model)


def plot(df):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns
    from utils import plt_style

    methods = ["CompoNet", "ProgressiveNet"]

    plt.rcParams.update({"font.size": 14})

    fig = plt.figure(figsize=(6, 5))

    # fmt: off
    g = sns.lineplot(
        data=df, x="num prevs", y="time",
        hue="method", style="method",
        markers=True, dashes=False,
        errorbar="sd", legend=False, hue_order=methods,
        linewidth=3, markersize=10,
    )
    # fmt: on

    # legend
    plt_style.style(fig, legend=False)
    lines = [Line2D([0], [0], color=c) for c in ["tab:blue", "tab:orange"]]
    fig.legend(
        lines,
        methods,
        fancybox=False,
        frameon=False,
        loc="outside lower center",
        ncols=2,
    )
    ax = plt.gca()
    p = 0.15  # %
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * p, box.width, box.height * (1 - p)])
    # labels
    plt.xlabel("Number of tasks")
    plt.ylabel("Inference time in seconds")

    plt.savefig("benchmarking_inference_time.pdf")
    # plt.show()

    plt.gca()
    fig = plt.figure(figsize=(6, 5))
    # fmt: off
    g = sns.lineplot(
        data=df, x="num prevs", y="total parameters",
        hue="method", style="method",
        markers=True, dashes=False,
        errorbar="sd", legend=False, hue_order=methods,
        linewidth=3, markersize=10,
    )
    g = sns.lineplot(
        data=df, x="num prevs", y="trainable parameters",
        hue="method", style="method",
        markers=True, dashes=False,
        errorbar="sd", legend=False, hue_order=methods,
        linewidth=1, markersize=7,
        linestyle="dashed",
    )
    # fmt: on

    plt_style.style(fig, legend=False, force_sci_y=True)

    lines = [Line2D([0], [0], color=c) for c in ["tab:blue", "tab:orange"]]
    lines += [
        Line2D([0], [0], color="black"),
        Line2D([0], [0], color="black", linestyle="dashed"),
    ]

    lbls = methods + ["Total", "Trainable"]
    lbls = [lbls[i] for i in [0, 2, 1, 3]]
    lines = [lines[i] for i in [0, 2, 1, 3]]

    fig.legend(
        lines, lbls, fancybox=False, frameon=False, loc="outside lower center", ncols=2
    )
    ax = plt.gca()
    p = 0.21  # %
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * p, box.width, box.height * (1 - p)])
    # labels
    plt.xlabel("Number of tasks")
    plt.ylabel("Number of parameters")
    plt.savefig("benchmarking_num_parameters.pdf")
    plt.show()


def plot_times(df, ax, fig, methods):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns
    from utils import plt_style

    g = sns.lineplot(
        data=df,
        x="num prevs",
        y="time",
        hue="method",
        style="method",
        markers=True,
        dashes=False,
        errorbar="sd",
        legend=False,
        hue_order=methods,
        linewidth=3,
        markersize=10,
        ax=ax,
    )
    ax.set_xticks(list(df["num prevs"].unique()))
    # fmt: on

    plt_style.style(fig, ax=ax, legend=False)

    # labels
    ax.set_xlabel("Number of tasks")
    ax.set_ylabel("Inference time in seconds")


def plot_memory(df, ax, fig, methods):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns
    from utils import plt_style

    # fmt: off
    g = sns.lineplot(
        data=df, x="num prevs", y="trainable parameters",
        hue="method", style="method",
        markers=True, dashes=False,
        errorbar="sd", legend=False, hue_order=methods,
        linewidth=1, markersize=7,
        linestyle="dashed", ax=ax,
    )
    g = sns.lineplot(
        data=df, x="num prevs", y="total parameters",
        hue="method", style="method",
        markers=True, dashes=False,
        errorbar="sd", legend=False, hue_order=methods,
        linewidth=3, markersize=10, ax=ax,
    )
    ax.set_xticks(list(df["num prevs"].unique()))
    # fmt: on

    plt_style.style(fig, ax=ax, legend=False, force_sci_y=True)
    ax.set_yscale("log")

    f = lambda x: x * 4 / 1e6
    axis_color = "lightgrey"
    sa = g.secondary_yaxis("right", functions=(f, f), color="white")
    sa.tick_params(colors="black", which="both")

    sa.set_ylabel("Memory in MB", color="black")
    from matplotlib.ticker import ScalarFormatter

    sa.yaxis.set_major_formatter(ScalarFormatter())
    sa.ticklabel_format(style="sci", axis="y", scilimits=(2, 2))
    sa.ticklabel_format(useMathText=True)

    lines = [
        Line2D([0], [0], color="black", linestyle="dashed"),
        Line2D([0], [0], color="black"),
    ]
    lbls = ["Trainable", "Total"]
    ax.legend(lines, lbls, fancybox=False, frameon=False)

    # labels
    ax.set_xlabel("Number of tasks")
    ax.set_ylabel("Number of parameters (log)")
    ax.set_ylim(1e5, 1e10)


def plot_joined(df):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    methods = ["CompoNet", "ProgressiveNet"]

    plt.rcParams.update({"font.size": 14})

    # fmt: off
    fig, axs = plt.subplots(
        ncols=2, nrows=1, figsize=(7, 5),
        layout='constrained', gridspec_kw={'wspace': 0.4, 'hspace': 0.2}
    )

    p = 0.2

    ax = axs[0]
    plot_times(df, ax, fig, methods)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * p, box.width, box.height * (1 - p)])

    ax = axs[1]
    plot_memory(df, ax, fig, methods)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * p, box.width, box.height * (1 - p)])

    # main legend (in the bottom of the figure)
    lines = [Line2D([0], [0], color=c, marker=m, linewidth=3,
                    markersize=10, markeredgecolor="white")
             for c, m in zip(["tab:blue", "tab:orange"], ["X", "o"])]
    fig.legend(
        lines, methods, fancybox=False, frameon=False, loc="outside lower center", ncols=2
    )

    plt.savefig("benchmarking.pdf", pad_inches=0.02, bbox_inches="tight")
    plt.show()


def plot_separate(df):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    methods = ["CompoNet", "ProgressiveNet"]
    plt.rcParams.update({"font.size": 14})

    def make_legend(fig, ax, p=0.2):
        # shrink
        box = ax.get_position()
        ax.set_position(
            [box.x0, box.y0 + box.height * p, box.width, box.height * (1 - p)]
        )
        # draw legend
        lines = [
            Line2D(
                [0],
                [0],
                color=c,
                marker=m,
                linewidth=3,
                markersize=10,
                markeredgecolor="white",
            )
            for c, m in zip(["tab:blue", "tab:orange"], ["X", "o"])
        ]
        fig.legend(
            lines,
            methods,
            fancybox=False,
            frameon=False,
            loc="outside lower center",
            ncols=2,
        )

    # Inference time
    fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()

    plot_times(df, ax, fig, methods)
    make_legend(fig, ax)

    plt.savefig("benchmarking_inference_time.pdf", pad_inches=0.02, bbox_inches="tight")
    plt.show()

    # Memory cost
    fig = plt.figure(figsize=(7, 7))
    ax = plt.gca()

    plot_memory(df, ax, fig, methods)
    make_legend(fig, ax, p=0.1)

    plt.savefig("benchmarking_memory.pdf", pad_inches=0.02, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    args = parse_args()

    if args.plot:
        if args.joined:
            plot_joined(pd.read_csv(args.save_path))
        else:
            plot_separate(pd.read_csv(args.save_path))
        quit()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.rand((args.batch_size, args.obs_dim)).to(device)

    dfs = []
    for num_prevs in args.test_prevs:
        print("\n==> Num. prevs.:", num_prevs)
        prognet = build_prognet(
            args.obs_dim, args.act_dim, args.hidden_dim, num_prevs, device
        )
        p_trainable = trainable_params(prognet)
        p_total = prognet_total_parameters(prognet)
        print(f"ProgressiveNet parameters: {p_total}, trainable: {p_trainable}")
        t0 = benchmark.Timer(
            stmt="model(x)",
            globals={"x": x, "model": prognet},
            num_threads=torch.get_num_threads(),
            description="ProgressiveNet",
            sub_label=str(num_prevs),
        )

        r = t0.blocked_autorange(min_run_time=args.min_run_time)
        df = pd.DataFrame(
            {
                "time": r.times,
                "method": "ProgressiveNet",
                "total parameters": p_total,
                "trainable parameters": p_trainable,
                "num prevs": num_prevs,
            }
        )
        dfs.append(df)

        compo = build_componet(
            args.obs_dim, args.act_dim, args.hidden_dim, num_prevs, device
        )
        p_trainable = trainable_params(compo)
        p_total = componet_total_parameters(compo)
        print(f"CompoNet parameters: {p_total}, trainable: {p_trainable}")
        t1 = benchmark.Timer(
            stmt="model(x)",
            globals={"x": x, "model": compo},
            num_threads=torch.get_num_threads(),
            description="CompoNet",
            sub_label=str(num_prevs),
        )

        r = t1.blocked_autorange(min_run_time=args.min_run_time)
        df = pd.DataFrame(
            {
                "time": r.times,
                "method": "CompoNet",
                "total parameters": p_total,
                "trainable parameters": p_trainable,
                "num prevs": num_prevs,
            }
        )
        dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv(args.save_path)
