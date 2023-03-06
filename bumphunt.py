import os
from copy import deepcopy
from imports import *
from utils import *
from data import *
from args import *
from collections import OrderedDict

def make_1d(events_sep, vname, nevents_sep, xsecs_sep, lumi, xmin, xmax, bins):
    xbins = np.linspace(xmin,xmax,bins+1)

    hist_output = {}
    for pkey, pevents in events_sep.items():
        weight = lumi*xsecs_sep[pkey]/nevents_sep[pkey]
        vals = np.asarray(pevents[vname])
        weights = weight*np.ones(shape=vals.shape)
        if hasattr(pevents,"Weight"):
            weights = weight*pevents.Weight
        counts, bins = np.histogram(vals, bins=xbins, weights=weights)
        hist_output["{}_{}".format(vname,pkey)] = {"counts": list(counts), "bins": list(bins)}

    return hist_output

def RT(events, mass):
    return events.Met.pt/mass

def event_selection(events, massname, mass, cut=None, verbose=False):
    # before selection, for lumi*xsec/nevent weighting: should be sum(events.Weight) for QCD
    if hasattr(events,"Weight"):
        nevents = sum(events.Weight)
    else:
        nevents = len(events)

    minpt = 200
    mask = events.Jet2.pt > minpt
    indices = np.squeeze(np.indices(np.asarray(mask).shape))
    events = events[mask]
    mass = mass[mask]
    indices = indices[mask]

    # todo: if using non-flat qcd sample, need to apply manual weights here
    RTvals = RT(events, mass)
    if cut is None:
        eff = 0.99
        nbins = 1000
        counts, bins = np.histogram(np.asarray(RTvals), bins=nbins, weights=np.asarray(events.Weight))
        counts = counts/np.sum(counts)
        cdf = np.cumsum(counts)
        idx = (np.abs(cdf-eff)).argmin()
        cut = bins[idx+1]
        if verbose: print("MET/mass cut = {}".format(cut))

    mask = RTvals > cut
    events = events[mask]
    mass = mass[mask]
    indices = indices[mask]
    # attach at the end to avoid out of memory / bad alloc
    events["RT"] = RT(events, mass)
    events[massname] = mass
    return events, nevents, cut, indices

def inference(network, vals, batch_size):
    outputs = None
    for val in np.array_split(vals, int(np.ceil(len(vals)/batch_size))):
        output = network(val).numpy()
        if outputs is None: outputs = output
        else: outputs = np.concatenate([outputs,output])
    return outputs

# uses separate datasets from training w/ distinct parameter values
def runner(argv, cut=None):
    eparser = EVNParser("runner")
    parser = eparser.parser
    parser.add_argument("--calib-source", type=str, default="", help="source of calibration coefficients")
    parser.add_argument("--uncalibrated", default=False, action="store_true", help="plot uncalibrated variables")
    parser.add_argument("--xmin", type=float, required=True, help="minimum for histogram x axis")
    parser.add_argument("--xmax", type=float, required=True, help="maximum for histogram x axis")
    parser.add_argument("--bins", type=int, default=100, help="number of bins for histogram")
    parser.add_argument("--lumi", type=float, default=138000, help="integrated luminosity [pb-1]")
    parser.add_argument("--batch-size", type=int, default=100000, help="batch size for inference")
    args = eparser.parse_args(argv=argv)
    outf_models = args.outf+"/"+args.model_dir

    # get physics process & data (as separate files)
    process = eparser.get_process(args)
    if len(process.theory)>1:
        raise ValueError("bumphunt only considers one variable at a time")
    events = process.get_events_sep()

    # convert to numpy format
    inputs = process.get_inputs(events)
    params = process.get_params(events)
    theory = process.get_theory(events)

    # empty theory implies AEV
    artvar = None
    if len(process.theory)==0:
        # get saved model
        model = import_attrs(args.model_dir, "AEVNetwork")(0, 0, [], "", outf_models)

        # compute the machine-learned variable
        artvar = OrderedDict([
            (key, inference(model.network, val, args.batch_size)) for key,val in inputs.items()
        ])

        massvals = artvar
        vname = "AEV"
    else:
        massvals = theory
        vname = process.theory[0]

    # perform calibration (includes flip)
    if not args.uncalibrated and len(args.calib_source)==0:
        raise ValueError("must provide calib_source if not uncalibrated")
    calibrations = import_attrs(args.calib_source, ["calibrations"])
    massvals = OrderedDict([(key,np.transpose(val)) for key,val in massvals.items()])
    vinfo = {"vals_sep": OrderedDict([(key,val[0]) for key,val in massvals.items()])}
    _, _, vinfo["vals_sep"] = calibrate(vinfo, [], coeff=calibrations[vname])
    massvals = vinfo["vals_sep"]

    # apply selection
    nevents = {}
    indices = {}
    for pkey, pevents in events.items():
        nevents[pkey] = 0
        events[pkey], nevents[pkey], cut, indices[str(pkey)] = event_selection(pevents, vname, massvals[pkey], cut, args.verbose)

    # make overlaid histograms
    hist_output = make_1d(events, vname, nevents, process.xsecs_sep, args.lumi, args.xmin, args.xmax, args.bins)

    # todo: 2D histograms: MET vs. mass, MET/mass vs. mass, MET/mass vs. deta

    return cut, hist_output, args, indices

def bumphunt():
    from magiconfig import ArgumentParser, ArgumentDefaultsRawHelpFormatter
    from argparse import _HelpAction

    # pass help to secondary parser
    class DelegatedHelpAction(_HelpAction):
        def __call__(self, parser, namespace, values, option_string=None):
            parser.print_help()
            print("")
            runner(["--help"])

    # simple parser that just gets a list of processes and then passes on the other arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsRawHelpFormatter, add_help=False)
    parser.add_argument("--namepre", type=str, default="bumphunt", required=True, help="prefix for saved config filename and output subdirectory")
    parser.add_argument("--signals", type=str, default=[], nargs='+', required=True, help="signal process config files")
    parser.add_argument("--background", type=str, default="", required=True, help="background process config file")
    parser.add_argument("--cut", type=float, default=None, help="specify RT cut (instead of deriving automatically)")
    parser.add_argument("-h", "--help", default=False, action=DelegatedHelpAction, help="show this help message and exit")
    args, unknown = parser.parse_known_args()
    outf_hist = None

    cut = args.cut
    hist_output = {}
    for process in [args.background]+args.signals:
        ctmp, htmp, rargs, indices = runner(unknown+["-C",process,"--name","{}_{}".format(args.namepre,os.path.basename(process).replace(".py",""))], cut)
        # get cut from background, then apply to MC
        if cut is None:
            cut = ctmp
        # get outf from runner (laziness)
        if outf_hist is None:
            outf_hist = rargs.outf+"/"+args.namepre
        hist_output.update(htmp)
        # save selected event indices
        np.savez("{}/{}_indices.npz".format(outf_hist,rargs.process), **indices)

    os.makedirs(outf_hist, exist_ok=True)
    with open("{}/hists.py".format(outf_hist), 'w') as hfile:
        hfile.write("hists = "+repr(hist_output))

if __name__=="__main__":
    bumphunt()
