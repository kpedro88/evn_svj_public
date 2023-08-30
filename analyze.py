import os
from copy import deepcopy
from imports import *
from utils import *
from data import *
from args import *
from collections import OrderedDict
from itertools import combinations
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.colors

def get_var_info(axes, names, values):
    var_info = OrderedDict()
    values = OrderedDict([(key,np.transpose(val)) for key,val in values.items()])
    for iname,name in enumerate(names):
        var_info[name] = deepcopy(axes[name])
        var_info[name]["name"] = name
        var_info[name]["vals_sep"] = OrderedDict([(key,val[iname]) for key,val in values.items()])
        var_info[name]["vals"] = np.concatenate([val[iname] for key,val in values.items()]) if values else np.array([])
    return var_info

def calibrate_sep(x, y, folder):
    coeff, fit, fit_sep = calibrate(x, y)
    fitlabel = ["fit (mx+b)"]
    coefflabels = ["b","m"]
    if len(coeff)==1:
        fitlabel = ["fit (mx)"]
        coefflabels = ["m"]
    fitlabel.extend(["{} = {:g}".format(clabel, cvalue) for cvalue, clabel in zip(coeff, coefflabels)])

    default_fontsize = plt.rcParams['font.size']
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(x["vals"], y["vals"], label="events")
    ax.plot(x["vals"], fit, label='\n'.join(fitlabel))
    ax.set_xlabel(x["label"], fontsize=default_fontsize+5)
    ax.set_ylabel(y["label"], fontsize=default_fontsize+5)
    ax.legend()
    figname = "calib_{}_{}".format(x["name"],y["name"])
    save_figure(plt, "{}/{}.pdf".format(folder,figname))

    # apply calibration to values
    x["vals"] = fit
    x["vals_sep"] = fit_sep

    return x

def make_labels(param_info):
    param_pref = ""
    if len(param_info["leg"])>0: param_pref = " = "
    param_suff = ""
    param_leg = param_info["leg"][:]
    param_unit = param_info["unit"] if "unit" in param_info else ""
    if param_leg[-1]=="$":
        param_leg = param_leg[:-1]
        if len(param_unit)>0: param_suff = r"\,\mathrm{"+param_unit+"}$"
        else: param_suff = "$"
    else:
        if len(param_unit)>0: param_suff = " "+param_unit
    return [param_leg+param_pref+str(pkey)+param_suff for pkey in param_info["vals_sep"]]

def make_1d(var_info, xmin, xmax, bins, xname, logy, extra_text, param_info, folder, index=None, res_table=False, weights=None):
    default_fontsize = plt.rcParams['font.size']
    fig, ax = plt.subplots(figsize=(9, 7))

    if xmin is None: xmin = np.min(np.concatenate([vinfo["vals"] for vname,vinfo in var_info.items()]))
    if xmax is None: xmax = np.max(np.concatenate([vinfo["vals"] for vname,vinfo in var_info.items()]))
    xbins = np.linspace(xmin,xmax,bins+1)

    line_list = get_lines()
    color_list = get_colors()
    label_list = make_labels(param_info)

    ctr = 0
    var_lines = []
    ymax = 0
    ymin = 1e10
    if res_table:
        headers = ["mass"]+[vname+" res" for vname in var_info]
        rows = [[] for _ in range(len(first(var_info)["vals_sep"]))]
        print("\t".join(headers))
    hist_output = {}
    for iline,(vname,vinfo) in enumerate(var_info.items()):
        if len(var_info)>1: var_lines.append(Line2D([0],[0],color='black',linestyle=line_list[iline],label=vinfo["leg"]))
        for icolor,(pkey,vals) in enumerate(vinfo["vals_sep"].items()):
            if index is not None and icolor!=index: continue
            weights_ = np.squeeze(weights[pkey]) if weights is not None else None
            counts, bins = np.histogram(vals, bins=xbins, weights=weights_, density=True)
            hist_output["{}_{}".format(vname,pkey)] = {"counts": list(counts), "bins": list(bins)}
            ax.hist(bins[:-1], bins, weights=counts, range=(xmin,xmax), histtype='step', linestyle=line_list[iline], color=color_list[icolor], label=label_list[icolor] if ctr==0 else '')
            ymax = max(ymax,max(counts))
            ymin = min(ymin,min(counts[counts>0]))
            if res_table:
                if iline==0:
                    rows[icolor].append(pkey)
                rows[icolor].append(np.std(vals)/np.mean(vals))
        ctr += 1
    if res_table:
        for row in rows:
            print("{}\t".format(row[0])+"\t".join(["{:.2f}".format(res) for res in row[1:]]))

    handles, labels = ax.get_legend_handles_labels()
    handles.extend(var_lines)

    if len(extra_text)>0:
        extra_handles = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=text) for text in extra_text]
        handles.extend(extra_handles)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    if logy:
        ax.set_yscale('log')
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        y_formatter = ScalarFormatterForceFormat('%1.1f')
        y_formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(y_formatter)
        ymin = 0

    ax.set_ylim(ymin, ymax*1.4)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(xname if len(var_info)>1 else list(var_info.items())[0][1]["label"], fontsize=default_fontsize+5)
    ax.set_ylabel("arbitrary units")
    ax.legend(handles=handles)

    save_figure(plt, "{}/hist1d{}.pdf".format(folder, "_{}".format(list(first(var_info)["vals_sep"].keys())[index]) if index is not None else ""))
    if index is None:
        with open("{}/hists.py".format(folder), 'w') as hfile:
            hfile.write("hists = "+repr(hist_output))

def make_2d(var_info, bins, xmin, xmax, ymin, ymax, extra_text, param_info, folder, index=None, logz=True, weights=None):
    default_fontsize = plt.rcParams['font.size']
    fig, ax = plt.subplots(figsize=(9, 7))

    x_info = nth(var_info,0)
    y_info = nth(var_info,1)
    if xmin is None: xmin = np.min(x_info["vals"])
    if xmax is None: xmax = np.max(x_info["vals"])
    if ymin is None: ymin = np.min(y_info["vals"])
    if ymax is None: ymax = np.max(y_info["vals"])

    color_list = get_colors()
    label_list = make_labels(param_info)

    alpha_max = 1.0
    exclude_pct = 0.0
    sep_lines = []
    for icolor in range(len(label_list)):
        if index is not None and icolor!=index: continue
        weights_ = np.squeeze(nth(weights,icolor)) if weights is not None else None
        x_vals = nth(x_info["vals_sep"],icolor)
        y_vals = nth(y_info["vals_sep"],icolor)
        # custom color map
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white",color_list[icolor]])
        fading_cmap = cmap(np.arange(cmap.N)) # extract colors
        exclude_N = int(cmap.N*exclude_pct)
        fading_cmap[:, -1] = np.concatenate((np.linspace(0, alpha_max, cmap.N-exclude_N),alpha_max*np.ones(exclude_N))) # modify alpha
        fading_cmap = matplotlib.colors.ListedColormap(fading_cmap)
        _ = ax.hist2d(x_vals, y_vals, bins=(bins, bins), range = [[xmin, xmax],[ymin, ymax]], weights=weights_, rasterized=True, cmap=fading_cmap, norm=mpl.colors.LogNorm() if logz else mpl.colors.Normalize())
        sep_lines.append(Rectangle((0, 0), 1, 1, facecolor=color_list[icolor], fill=True, edgecolor='none', linewidth=0, label=label_list[icolor]))

    handles, labels = ax.get_legend_handles_labels()
    handles.extend(sep_lines)
    if len(extra_text)>0:
        extra_handles = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=text) for text in extra_text]
        handles.extend(extra_handles)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(x_info["label"], fontsize=default_fontsize+5)
    ax.set_ylabel(y_info["label"], fontsize=default_fontsize+5)
    ax.legend(handles=handles)

    save_figure(plt, "{}/scatter_{}_{}{}.pdf".format(folder, list(var_info.keys())[0], list(var_info.keys())[1], "_{}".format(list(first(var_info)["vals_sep"].keys())[index]) if index is not None else ""))

def pairing(extras, target, pkey=None, outputs=[]):
    dkey = "vals" if pkey is None else "vals_sep"
    inputs = [
        extras["Mjj_msortedP1_high"][dkey],
        extras["Mjj_msortedP1_low"][dkey],
        extras["Mjj_msortedP2_high"][dkey],
        extras["Mjj_msortedP2_low"][dkey],
        extras["Mjj_msortedP3_high"][dkey],
        extras["Mjj_msortedP3_low"][dkey],
    ]
    if pkey is not None: inputs = [item[pkey] for item in inputs]
    metric = np.array([
        (inputs[0]-target)**2+(inputs[1]-target)**2,
        (inputs[2]-target)**2+(inputs[3]-target)**2,
        (inputs[4]-target)**2+(inputs[5]-target)**2,
    ])
    best = np.argmin(metric,axis=0)
    if len(outputs)==0:
        return best
    outputs = {key:[] for key in outputs}
    for key in outputs.keys():
        if key=="avg":
            mout = np.array([
                (inputs[0]+inputs[1])/2.,
                (inputs[2]+inputs[3])/2.,
                (inputs[4]+inputs[5])/2.,
            ])
        elif key=="high":
            mout = np.array([
                inputs[0],
                inputs[2],
                inputs[4],
            ])
        elif key=="low":
            mout = np.array([
                inputs[1],
                inputs[3],
                inputs[5],
            ])
        outputs[key] = np.take_along_axis(mout,best[None],axis=0).T
    return best, outputs

# uses separate datasets from training w/ distinct parameter values
def analyze():
    eparser = EVNParser("plots")
    parser = eparser.parser
    parser.add_argument("--calib-source", type=str, default="", help="source of calibration coefficients")
    parser.add_argument("--uncalibrated", default=False, action="store_true", help="plot uncalibrated variables (skips AEV)")
    parser.add_argument("--xmin", type=float, default=None, help="minimum for histogram x axis (if not auto)")
    parser.add_argument("--xmax", type=float, default=None, help="maximum for histogram x axis (if not auto)")
    parser.add_argument("--bins", type=int, default=100, help="number of bins for histogram")
    parser.add_argument("--xname", type=str, default="mass [calibrated]", help="label for histogram x axis")
    parser.add_argument("--ymin", type=float, default=None, help="minimum for scatterplot y axis (if not auto)")
    parser.add_argument("--ymax", type=float, default=None, help="maximum for scatterplot y axis (if not auto)")
    parser.add_argument("--logy", default=False, action="store_true", help="logarithmic y axis")
    parser.add_argument("--extra-text", type=str, default=[], action="append", help="extra text for legend (can be called multiple times)")
    parser.add_argument("--npz", type=str, default=None, help="load an npz file of test event indices")
    parser.add_argument("--pair", default=False, action="store_true", help="compute pairing accuracy")
    parser.add_argument("--pair-mass", default=False, action="store_true", help="plot mass using predicted pairing")
    args = eparser.parse_args()
    if args.extra_text and not isinstance(args.extra_text,list): args.extra_text = [args.extra_text]
    outf_models = args.outf+"/"+args.model_dir
    outf_test = args.outf+"/"+args.name
    os.makedirs(outf_test, exist_ok=True)

    plot_format()

    # get physics process & data (as separate files)
    process = eparser.get_process(args)
    events = process.get_events_sep()

    # import list of test events
    event_list = None
    if args.npz is not None:
        event_list = np.load(args.npz)

    # convert to numpy format
    inputs = process.get_inputs(events,event_list)
    params = process.get_params(events,event_list)
    theory = process.get_theory(events,event_list)
    weights = process.get_weights(events,event_list)

    # computed automatically by process
    if not args.uncalibrated:
        # get saved model
        model = import_attrs(args.model_dir, "AEVNetwork")(0, 0, [], "", outf_models)

        # compute the machine-learned variable
        artvars = OrderedDict([
            (key, model.network(val).numpy()) for key,val in inputs.items()
        ])

    # labels taken from axis info input
    var_params = first(get_var_info(args.axes, list(process.params), params))
    var_info = OrderedDict()
    if not args.uncalibrated:
        n_artvars = first(artvars).shape[1]
        names_artvars = []
        artname = "AEV"
        for col in range(n_artvars):
            artvar = OrderedDict([(key, val[:,col][:,None]) for key,val in artvars.items()])
            if n_artvars>1: artname = "AEV{}".format(col)
            var_info[artname] = first(get_var_info(args.axes, [artname], artvar))
            names_artvars.append(artname)
    if theory is not None: var_info.update(get_var_info(args.axes, process.theory, theory))

    # perform calibration
    if not args.uncalibrated:
        if len(args.calib_source)==0:
            for col in range(n_artvars):
                if n_artvars>1: artname = "AEV{}".format(col)
                # Ensuring positive correlation
                _, _, flip = correl(var_info[artname]["vals"], var_params["vals"])
                if flip:
                    var_info[artname]["vals_sep"] = OrderedDict([(key,-1*val) for key,val in var_info[artname]["vals_sep"].items()])
                    var_info[artname]["vals"] *= -1
            for vname,vinfo in var_info.items():
                var_info[vname] = calibrate_sep(vinfo, var_params, outf_test)
        else:
            # flip included in calibration
            calibrations = import_attrs(args.calib_source, ["calibrations"])
            for vname,vinfo in var_info.items():
                _, vinfo["vals"], vinfo["vals_sep"] = calibrate(vinfo, var_params, coeff=calibrations[vname])

    # get derived mass from pairing (no calib)
    if args.pair_mass:
        extras_vals = process.get_extras(events,event_list)
        extras = get_var_info(args.axes, process.extras, extras_vals)
        pm_vals = OrderedDict()
        for pkey,vals in var_info["AEV"]["vals_sep"].items():
            _, predM = pairing(extras, vals, pkey, outputs=["avg"])
            pm_vals[pkey] = [predM["avg"]]
        var_info.update(get_var_info(args.axes, ["pair_M_avg"], pm_vals))

    # make overlaid histogram
    make_1d(var_info, args.xmin, args.xmax, args.bins, args.xname, args.logy, args.extra_text, var_params, outf_test, res_table=args.verbose, weights=weights)

    # make separate histograms
    if theory is not None and len(theory)>1:
        for index in range(len(theory)):
            make_1d(var_info, args.xmin, args.xmax, args.bins, args.xname, args.logy, args.extra_text, var_params, outf_test, index, weights=weights)

    # 2D scatterplots for n_artvar>1 case
    if not args.uncalibrated and n_artvars>1:
        for comb in combinations(names_artvars,2):
            comb_info = OrderedDict([(comb_i, var_info[comb_i]) for comb_i in comb])
            make_2d(comb_info, args.bins, args.xmin, args.xmax, args.ymin, args.ymax, args.extra_text, var_params, outf_test, weights=weights)

    # pairings
    if args.pair:
        extras_vals = process.get_extras(events,event_list)
        extras = get_var_info(args.axes, process.extras, extras_vals)
        masses = np.concatenate([pkey*np.ones_like(vals) for pkey,vals in var_info["AEV"]["vals_sep"].items()])
        truth = pairing(extras, masses)
        # minimize 1 - acc
        def acc(x):
            pred = pairing(extras, x[0]*var_info["AEV"]["vals"])
            target = 1 - sum(truth==pred)/len(var_info["AEV"]["vals"])
            return target
        from scipy.optimize import minimize
        # full calibration is x * initial calibration
        x0 = np.array([1])
        res = minimize(acc, x0, method='nelder-mead', options={'disp': args.verbose})
        x1 = res.x[0]
        masses = []
        accs = []
        for pkey,vals in var_info["AEV"]["vals_sep"].items():
            truth = pairing(extras, pkey, pkey)
            pred = pairing(extras, x1*vals, pkey)
            masses.append(pkey)
            accs.append(sum(truth==pred)/len(vals))
        acc_output = {'x': masses, 'y': accs}
        with open("{}/acc.py".format(outf_test), 'w') as afile:
            afile.write("acc = "+repr(acc_output))
        # update calibration
        new_calibrations = {key: 1.0 for key in calibrations.keys()}
        new_calibrations["AEV"] = [x1*calibrations["AEV"][0]]
        with open("{}/calibrations.py".format(outf_test), 'w') as cfile:
            cfile.write("calibrations = "+repr(new_calibrations))

if __name__=="__main__":
    analyze()
