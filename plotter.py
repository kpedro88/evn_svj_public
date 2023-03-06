import os
from imports import *
from utils import *
from collections import OrderedDict
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

def subplotter(ax, subcomponents):
    leg = subcomponents.pop("leg",{})
    do_leg = leg["on"] if leg else False
    leg_extra = []

    for cname, component in subcomponents.items():
        if cname=="x":
            if "name" in component: ax.set_xlabel(component["name"])
            if "range" in component: ax.set_xlim(*component["range"])
            if component.get("log",False): ax.set_xscale('log')
        elif cname=="y":
            if "name" in component: ax.set_ylabel(component["name"])
            if "range" in component: ax.set_ylim(*component["range"])
            if component.get("log",False): ax.set_yscale('log')
        elif "counts" in component:
            ax.hist(component["bins"][:-1], component["bins"], weights=component["counts"], histtype='step', linestyle=component["linestyle"], color=component["color"], label=component["leg"])
        elif do_leg:
            if "linestyle" in component:
                leg_extra.append(Line2D([0], [0], color=component["color"], linestyle=component["linestyle"], label=component["leg"]))
            else:
                leg_extra.append(Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=component["leg"]))

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    if do_leg:
        handles, labels = ax.get_legend_handles_labels()
        handles.extend(leg_extra)
        ax.legend(handles=handles,**leg.pop("kwargs",{}))

    return ax

def plotter(components, widths, heights, folder, outname):
    plot_format()
    default_fontsize = plt.rcParams['font.size']

    # check consistency
    nrows = len(components)
    ncols = [len(components[i]) for i in range(len(components))]
    if len(set(ncols))>1:
        raise ValueError("Inconsistent second dimension in components: values are {}".format(ncols))
    ncols = ncols[0]

    subplot_args = {"ncols": ncols, "nrows": nrows, "figsize": (sum(widths),sum(heights))}
    gridspec_kw = {}
    if nrows>1:
        subplot_args.update({"sharex": True})
        gridspec_kw.update({"height_ratios": heights})
    if ncols>1:
        subplot_args.update({"sharey": True})
        gridspec_kw.update({"width_ratios": widths})
    if len(gridspec_kw)>0:
        subplot_args.update({"gridspec_kw": gridspec_kw})
    fig, axs = plt.subplots(**subplot_args)
    if not isinstance(axs,np.ndarray): axs = np.asarray([[axs]])
    elif nrows==1: axs = np.expand_dims(axs,0)
    elif ncols==1: axs = np.expand_dims(axs,1)

    for irow in range(nrows):
        for icol in range(ncols):
            axs[irow][icol] = subplotter(axs[irow][icol], components[irow][icol])

    save_figure(plt, "{}/{}.pdf".format(folder, outname))
