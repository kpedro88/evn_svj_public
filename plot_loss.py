import os
from imports import *
from utils import *
from args import *
from models import BasicNetwork
from matplotlib.lines import Line2D

def get_loss_path(folder):
    model = BasicNetwork(None)
    return os.path.dirname(model.filename(folder))

def plot_loss():
    eparser = EVNParser("loss",basic=True)
    parser = eparser.parser
    parser.add_argument("--curves", type=str, default=None, help=".py file w/ OrderedDict of curve info (if empty, just plots loss from outf)")
    parser.add_argument("--yrange", metavar=("ymin","ymax"), type=float, default=None, nargs=2, help="y axis range (None=auto)")
    parser.add_argument("--val", default=False, action="store_true", help="include validation lines (for multiple curve mode)")
    parser.add_argument("--extend", default=False, action="store_true", help="extend shorter lines to maximum epoch")
    args = eparser.parse_args()
    outf_loss = args.outf+"/"+args.name
    os.makedirs(outf_loss, exist_ok=True)

    plot_format()

    default_fontsize = plt.rcParams['font.size']
    fig, ax = plt.subplots(figsize=(9, 7))

    line_list = get_lines()
    color_list = get_colors()
    if args.curves is not None:
        curves = import_attrs(args.curves, "curves")
        var_lines = []
        max_len = 0
        for cname, curve in curves.items():
            losses = None
            epoch_sel = 0
            used_best = False
            for ipath,path in enumerate(curve["folder"]):
                path_models = path+"/"+args.model_dir
                loss_path = get_loss_path(path_models)
                losses_tmp = np.load("{}/loss.npz".format(loss_path))['arr_0']
                epoch_path = "{}/epoch.txt".format(loss_path)
                epoch_tmp = None
                if os.path.isfile(epoch_path):
                    used_best = True
                    with open(epoch_path,'r') as efile:
                        epoch_tmp = int(efile.read().rstrip().split()[0])
                if epoch_tmp is None: epoch_tmp = len(losses_tmp)-1
                if ipath<len(curve["folder"])-1: losses_tmp = losses_tmp[:, :epoch_tmp+1]
                epoch_sel += epoch_tmp
                if ipath>0: epoch_sel += 1
                # allow concatenating losses from continued training runs
                if losses is None:
                    losses = losses_tmp
                else:
                    losses = np.concatenate([losses, losses_tmp],axis=1)
            curve["array"] = losses
            if used_best: curve["epoch"] = epoch_sel
            max_len = max(max_len,losses.shape[1])
        for icolor, (cname, curve) in enumerate(curves.items()):
            losses = curve["array"]
            ax.plot(losses[0], color=color_list[icolor], linestyle="solid", label=curve["leg"])
            if args.extend and losses.shape[1] < max_len:
                extend_x = np.arange(losses.shape[1],max_len)
                extend_y = losses[0][-1]*np.ones(len(extend_x))
                ax.plot(extend_x, extend_y, color=color_list[icolor], linestyle="dotted")
            if args.val:
                ax.plot(losses[1], color=color_list[icolor], linestyle="dashed")
                if args.extend and losses.shape[1] < max_len:
                    extend_y = losses[1][-1]*np.ones(len(extend_x))
                    ax.plot(extend_x, extend_y, color=color_list[icolor], linestyle="dashdot")
            if "epoch" in curve:
                ax.plot([curve["epoch"]], [losses[1][curve["epoch"]]], color=color_list[icolor], marker='o', markerfacecolor='black')
        if args.val:
            var_lines.append(Line2D([0],[0],color='black',linestyle="solid",label="training"))
            var_lines.append(Line2D([0],[0],color='black',linestyle="dashed",label="validation"))
        if any("epoch" in curve for curve in curves.values()):
            var_lines.append(Line2D([0],[0],color='black',linestyle='None',markerfacecolor='black',marker='o',label="selected"))
        handles, labels = ax.get_legend_handles_labels()
        handles.extend(var_lines)
        ax.legend(handles=handles)
    else:
        loss_path = get_loss_path(args.outf+"/"+args.model_dir)
        losses = np.load("{}/loss.npz".format(loss_path))['arr_0']
        ax.plot(losses[0], color=color_list[1], linestyle="solid", label="training")
        ax.plot(losses[1], color=color_list[2], linestyle="dashed", label="validation")
        epoch_path = "{}/epoch.txt".format(loss_path)
        if os.path.isfile(epoch_path):
            with open(epoch_path,'r') as efile:
                epoch = int(efile.read().rstrip().split()[0])
                ax.plot([epoch],[losses[1][epoch]], color='black', marker='o', markerfacecolor='black', label="best")
        ax.legend()

    if args.yrange is not None:
        ax.set_ylim(args.yrange[0], args.yrange[1])
    ax.set_xlabel("epoch", fontsize=default_fontsize+5)
    ax.set_ylabel("loss", fontsize=default_fontsize+5)
    save_figure(plt, "{}/{}.pdf".format(outf_loss,"losses"))

if __name__=="__main__":
    plot_loss()
