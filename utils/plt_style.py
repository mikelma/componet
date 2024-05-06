import matplotlib.pyplot as plt
import matplotlib
import colormaps as cmaps


def style(fig, ax=None, grid=True, legend=True, legend_title=None, legend_ncols=1, force_sci_x=False, force_sci_y=False, font=3, colormap=cmaps.greenorange_12, legend_shrink=0.1, ax_math_ticklabels=True, y_spine=False):
    if colormap is not None:
        plt.set_cmap(colormap)

    # remove lateral spines
    ax = ax if ax is not None else plt.gca()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(y_spine)
    if ax_math_ticklabels:
        ax.ticklabel_format(useMathText=True)

    # axis sci notation
    if force_sci_x or force_sci_y:
        ax.ticklabel_format(useOffset=False)
        ax.ticklabel_format(style='sci',
                            axis='x' if force_sci_x else 'y',
                            scilimits=(0,0))

    # set spine and tick width and color
    axis_color = "lightgrey"
    ax.spines["bottom"].set(linewidth=1.3, color=axis_color)
    ax.spines["left"].set(linewidth=1.3, color=axis_color)
    ax.xaxis.set_tick_params(width=1.3, color=axis_color)
    yc = axis_color if y_spine else "white"
    ax.yaxis.set_tick_params(width=1.3, color=yc)

    if legend:
        l = fig.legend(title=legend_title, fancybox=False, frameon=False, loc="outside lower center", ncols=legend_ncols)
        # Shrink current axis's height by 10% on the bottom
        p = legend_shrink # %
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * p,
                         box.width, box.height * (1-p)])

    if grid:
        ax.grid(True, axis='y', alpha=0.2, linestyle='-')
        ax.yaxis.set_tick_params(size=0)
