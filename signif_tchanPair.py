import numpy as np
from utils import import_attrs, get_lines, get_colors
from collections import OrderedDict
from copy import deepcopy
from plotter import plotter

lines = get_lines()
colors = get_colors()

dir = "tchanPair/"
hists = {}
vars = ["AEV","PairMT2Reco"]
for var in vars:
    hists.update(import_attrs(dir+"bumphunt_{}/hists.py".format(var),"hists"))

components = [
    [OrderedDict([ # events
        ("QCD_AEV", {"counts": hists["AEV_15"]["counts"], "bins": hists["AEV_15"]["bins"], "color": colors[0], "linestyle": lines[0], "leg": "QCD"}),
        ("M500_AEV", {"counts": hists["AEV_500"]["counts"], "bins": hists["AEV_500"]["bins"], "color": colors[1], "linestyle": lines[0], "leg": r"$m_{\Phi} = 500\,\mathrm{GeV}$"}),
        ("M1000_AEV", {"counts": hists["AEV_1000"]["counts"], "bins": hists["AEV_1000"]["bins"], "color": colors[2], "linestyle": lines[0], "leg": r"$m_{\Phi} = 1000\,\mathrm{GeV}$"}),
        ("M2000_AEV", {"counts": hists["AEV_2000"]["counts"], "bins": hists["AEV_2000"]["bins"], "color": colors[3], "linestyle": lines[0], "leg": r"$m_{\Phi} = 2000\,\mathrm{GeV}$"}),
        ("QCD_PairMT2Reco", {"counts": hists["PairMT2Reco_15"]["counts"], "bins": hists["PairMT2Reco_15"]["bins"], "color": colors[0], "linestyle": lines[1], "leg": ""}),
        ("M500_PairMT2Reco", {"counts": hists["PairMT2Reco_500"]["counts"], "bins": hists["PairMT2Reco_500"]["bins"], "color": colors[1], "linestyle": lines[1], "leg": ""}),
        ("M1000_PairMT2Reco", {"counts": hists["PairMT2Reco_1000"]["counts"], "bins": hists["PairMT2Reco_1000"]["bins"], "color": colors[2], "linestyle": lines[1], "leg": ""}),
        ("M2000_PairMT2Reco", {"counts": hists["PairMT2Reco_2000"]["counts"], "bins": hists["PairMT2Reco_2000"]["bins"], "color": colors[3], "linestyle": lines[1], "leg": ""}),
        ("AEV", {"linestyle": lines[0], "color": "black", "leg": r"$V$"}),
        ("PairMT2Reco", {"linestyle": lines[1], "color": "black", "leg": r"$M_{\mathrm{T2}}$"}),
        ("y", {"name": r"events [138 $\mathrm{fb}^{-1}$]", "log": True}),
        ("x", {"range": [0, 3000]}),
        ("leg", {"on": True}),
    ])],
    [OrderedDict([ # signif
        ("y", {"name": r"$S/\sqrt{B}$", "log": True, "range": [0.01,100]}),
    ])],
    [OrderedDict([ # signif_ratio
        ("line", {"counts": np.ones_like(hists["AEV_1000"]["counts"]), "bins": hists["AEV_1000"]["bins"], "color": "black", "linestyle": "dashed", "leg": ""}),
        ("x", {"name": "mass [calibrated]"}),
        ("y", {"name": r"$S/\sqrt{B}$ ratio ($V$/*)", "log": False, "range": [0, 3]}),
    ])],
]

for var in vars:
    for sig in ["M500","M1000","M2000"]:
        svname = "{}_{}".format(sig,var)
        components[1][0][svname] = deepcopy(components[0][0][svname])
        numer = components[1][0][svname]["counts"]
        denom = np.sqrt(components[0][0]["QCD_{}".format(var)]["counts"])
        components[1][0][svname]["counts"] = np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0)
        print("{} ({}): {:.4g}".format(var,sig,np.sqrt(np.sum(components[1][0][svname]["counts"]**2))))

        if var=="AEV": continue
        components[2][0][svname] = deepcopy(components[0][0][svname])
        numer = components[1][0][svname.replace(var,"AEV")]["counts"]
        denom = components[1][0][svname]["counts"]
        components[2][0][svname]["counts"] = np.divide(numer, denom, out=np.zeros_like(numer), where=((denom!=0) & (np.asarray(components[0][0][svname]["bins"][:-1]) < (float(sig[1:])+800))))

plotter(components, [9], [7,2,2], dir, "signif_tchanPair")
