import os
from copy import deepcopy
from imports import *
from utils import *
from data import *
from args import *
from collections import OrderedDict
from itertools import combinations

def heatmap(hists, name, folder, calibrations=None, logz=False, verbose=False, weights=None):
    default_fontsize = plt.rcParams['font.size']
    figwidth = 6
    fig, axs = plt.subplots(1,len(hists),figsize=(len(hists)*figwidth, 6))
    if not isinstance(axs,np.ndarray): axs = [axs]
    cb_formatter = ScalarFormatterForceFormat('%1.1f')
    cb_formatter.set_powerlimits((0, 0))

    if weights is not None: weights = np.squeeze(weights)
    for hist, ax in zip(hists,axs):
        _ = ax.hist2d(hist["x"]["vals"], hist["y"]["vals"], bins=(hist["x"]["bins"], hist["y"]["bins"]), range = [[hist["x"]["lims"][0],hist["x"]["lims"][1]],[hist["y"]["lims"][0],hist["y"]["lims"][1]]], weights=weights, density=True, rasterized=True, norm=mpl.colors.LogNorm() if logz else mpl.colors.Normalize())
        ax.set_xlabel(hist["x"]["label"], fontsize=default_fontsize+2)
        ax.set_ylabel(hist["y"]["label"], fontsize=default_fontsize+2)
        #ax.set_aspect(1/ax.get_data_ratio())
        fig.colorbar(_[3], ax=ax, format=None if logz else cb_formatter)

        if hist["correl"]:
            kendalltau, spearmanr, _ = correl(hist["x"]["vals"], hist["y"]["vals"])
            # get the calibration line for this combination
            coeff, _, _ = calibrate(hist["x"], hist["y"])
            skewness_val = skewness(hist["x"]["vals"], hist["y"]["vals"], coeff[0])
            t0 = ax.text(.97, .17, f"$\\gamma_1^{{\\mathrm{{rel}}}}={skewness_val:.3f}$", color='white', transform=ax.transAxes, va='bottom', ha='right', fontsize=default_fontsize+2)
            t1 = ax.text(.97, .10, f"$\\tau={kendalltau:.3f}$", color='white', transform=ax.transAxes, va='bottom', ha='right', fontsize=default_fontsize+2)
            t2 = ax.text(.97, .03, f"$r_s={spearmanr:.3f}$", color='white', transform=ax.transAxes, va='bottom', ha='right', fontsize=default_fontsize+2)
            if logz:
                for tt in [t0,t1,t2]:
                    tt.set_path_effects([path_effects.withStroke(linewidth=5, foreground='k')])

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    fig.tight_layout()
    save_figure(plt, "{}/{}.pdf".format(folder,name))

    if calibrations is not None:
        # add calibration lines to existing plots
        for hist, ax in zip(hists,axs):
            coefflabels = ["m","b"]
            fitlabel = "fit (m{}+b)"
            if "fit" in hist["x"] and "fit" not in hist["y"]:
                line_x = hist["x"]["vals"]
                line_y = hist["x"]["fit"]
                fitlabel = fitlabel.format("x")
                coeff = calibrations[hist["x"]["name"]]
            elif "fit" not in hist["x"] and "fit" in hist["y"]:
                line_x = hist["y"]["fit"]
                line_y = hist["y"]["vals"]
                fitlabel = fitlabel.format("y")
                coeff = calibrations[hist["y"]["name"]]
            else:
                if verbose: print("WARNING: cannot display calibration fit for {}, {}".format(x["name"], y["name"]))
                continue

            if len(coeff)==1:
                coefflabels = ["m"]
                fitlabel = fitlabel.replace("+b)",")")
            fitlabels = [fitlabel]
            fitlabels.extend(["{} = {:.3g}".format(clabel, cvalue) for cvalue, clabel in zip(coeff, coefflabels)])
            ax.plot(line_x, line_y, label='\n'.join(fitlabels), color='k')
            ax.legend()

        save_figure(plt, "{}/calib_{}.pdf".format(folder,name))

def get_axis_info(axes, names, values, pranges=None):
    axis_info = OrderedDict()
    if not isinstance(values,list):
        values = np.transpose(values)
    if pranges is None:
        pranges = [None]*len(names)
    else:
        # account for multiple ranges
        pranges = [(p[0][0],p[-1][-1]) if isinstance(p,list) else p for p in pranges]
    for name, value, prange in zip(names, values, pranges):
        axis_info[name] = deepcopy(axes[name])
        axis_info[name]["name"] = name
        if prange is not None: axis_info[name]["lims"] = prange
        axis_info[name]["vals"] = np.squeeze(value)
    return axis_info

def do_calibrate(name, axis, axis_params, calibrations):
    x = axis
    y = axis_params
    coeff, fit, _ = calibrate(axis, axis_params)
    calibrations[name] = coeff
    axis["fit"] = fit

def check_test_args(args):
    if len(args.outfs)>0:
        if (len(args.model_dirs)>0 and len(args.model_dirs)!=len(args.outfs)) or (len(args.inputs_multi)>0 and len(args.inputs_multi)!=len(args.outfs)):
            raise ValueError("Length of model_dirs ({}) and inputs_multi ({}) must match length of outfs ({}) or be zero".format(len(args.outfs),len(args.model_dirs),len(args.inputs_multi)))
    return args

# uses "test" portion of training dataset
# run separately because plots can be finicky
def test():
    eparser = EVNParser("test")
    parser = eparser.parser
    parser.add_argument("--calibrate", default=False, action="store_true", help="calibrate variables using test data")
    parser.add_argument("--npz", type=str, default="", help="load a different npz file of test event indices")
    parser.add_argument("--outfs", type=str, default=[], nargs='*', help="multiple output folders to compare different models")
    parser.add_argument("--model-dirs", type=str, default=[], nargs='*', help="directories for saved model files to compare different models (empty: use --model-dir for all)")
    parser.add_argument("--inputs-multi", type=str, default=[], nargs='*', action="append", help="input feature name(s) to compare different models (call once per model) (only if different from --inputs)")
    parser.add_argument("--flip", type=int, default=[], nargs='*', help="manually flip specified AEVs")
    parser.add_argument("--logz", default=False, action="store_true", help="logarithmic z axis (color scale)")
    args = eparser.parse_args(checker=check_test_args)
    outf_test = args.outf+"/"+args.name
    os.makedirs(outf_test, exist_ok=True)

    plot_format()

    # fit the single model case into the multi model case
    if len(args.outfs)==0: args.outfs = [args.outf]
    if len(args.model_dirs)==0: args.model_dirs = [args.model_dir]*len(args.outfs)
    if len(args.inputs_multi)==0: args.inputs_multi = [args.inputs]*len(args.outfs)

    # make superset of all inputs
    args.inputs = list(set([i for inputs_tmp in args.inputs_multi for i in inputs_tmp]))

    # get physics process & data
    process = eparser.get_process(args)
    events = process.get_events()

    # import list of test events (consistent w/ train.py)
    shuffler_path = args.npz if len(args.npz)>0 else "{}/shuffler_test.npz".format(args.outf+"/"+args.model_dir)
    shuffler_test = np.load(shuffler_path)
    if 'arr_0' in shuffler_test: shuffler_test = {"": shuffler_test['arr_0']}
    else: shuffler_test = {"": list(shuffler_test.values())[0]}

    # convert to numpy format
    params = process.get_params(events,shuffler_test)
    theory = process.get_theory(events,shuffler_test)
    extras = process.get_extras(events,shuffler_test)
    weights = process.get_weights(events,shuffler_test)

    # get saved model(s) & input(s)
    inputs = []
    models = []
    artvars = []
    for outf,model_dir,args_input in zip(args.outfs,args.model_dirs,args.inputs_multi):
        process.inputs = args_input
        inputs.append(process.get_inputs(events,shuffler_test))
        models.append(import_attrs(model_dir, "AEVNetwork")(0, 0, [], "", outf+"/"+model_dir))
        # compute the machine-learned variable(s)
        artvars.append(np.nan_to_num(models[-1].network(inputs[-1]).numpy(),posinf=0,neginf=0))

    # put all ML variables in single numpy array (same format as single ML model w/ bottleneck > 1)
    artvars = np.concatenate(artvars, axis=1)

    # histogram axis info, taken from config (mostly)
    if params is not None: axis_params = first(get_axis_info(args.axes, list(process.params), params, list(process.params.values())))
    if theory is not None: axis_theory = get_axis_info(args.axes, process.theory, theory)
    if extras is not None: axis_extras = get_axis_info(args.axes, process.extras, extras)

    calibrations = {}
    # theory vars get calibrated against param
    if args.calibrate:
        for name, axis in axis_theory.items():
            do_calibrate(name, axis, axis_params, calibrations)

    n_artvars = artvars.shape[1]
    artname = "AEV"
    axis_artvars = OrderedDict()
    for col in range(n_artvars):
        artvar = artvars[:,col]
        if n_artvars>1: artname = "AEV{}".format(col)

        # Ensuring positive correlation
        flip = False
        if params is not None:
            _, _, flip = correl(artvar, axis_params["vals"])
        if col in args.flip:
            flip = True
        if flip: artvar *= -1

        axis_artvar = first(get_axis_info(args.axes, [artname], [artvar], [(min(0, np.mean(artvar)-3*np.std(artvar)), np.mean(artvar)+3*np.std(artvar))]))
        axis_artvars[artname] = axis_artvar

        # AEV(s) get calibrated against param
        if args.calibrate:
            do_calibrate(artname, axis_artvar, axis_params, calibrations)
            # include flip in coeff for later use
            if flip: calibrations[artname][-1] *= -1

        # theory vars get plotted w/ param & AEV
        if theory is not None:
            for name, axis in axis_theory.items():
                heatmap(
                    hists = [
                        {"x": axis_params, "y": axis, "correl": False},
                        {"x": axis_params, "y": axis_artvar, "correl": False},
                        {"x": axis, "y": axis_artvar, "correl": True},
                    ],
                    name = "heatmap_{}_theory_{}".format(col,name),
                    folder = outf_test,
                    calibrations = calibrations if args.calibrate else None,
                    verbose = args.verbose,
                    weights = weights,
                    logz = args.logz,
                )

        # extra vars get plotted w/ theory var(s) & AEV
        if extras is not None:
            for name, axis in axis_extras.items():
                hists = []
                for name_th, axis_th in axis_theory:
                    hists.append({"x": axis, "y": axis_th, "correl": True})
                hists.append({"x": axis, "y": axis_artvar, "correl": True})
                heatmap(
                    hists = hists,
                    name = "heatmap_{}_extra_{}".format(col,name),
                    folder = outf_test,
                    weights = weights,
                    logz = args.logz,
                )

    # plot AEVs against each other
    if n_artvars>1:
        # one plot per x-axis variable (last key is never used for x axis)
        hist_groups = OrderedDict([(key,[]) for key in list(axis_artvars.keys())[:-1]])
        for comb in combinations(axis_artvars.keys(),2):
            hist_groups[comb[0]].append({"x": axis_artvars[comb[0]], "y": axis_artvars[comb[1]], "correl": True})
        for key,hists in hist_groups.items():
            heatmap(
                hists = hists,
                name = "heatmap_AEVs_vs_{}".format(key),
                folder = outf_test,
                weights = weights,
                logz = args.logz,
            )

    with open(outf_test+"/calibrations.py",'w') as cfile:
        calibrations = {name:list(coeff) for name,coeff in calibrations.items()}
        cfile.write("calibrations = "+repr(calibrations))

if __name__=="__main__":
    test()
