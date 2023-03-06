import numpy as np
from utils import import_attrs, get_lines, get_colors
from collections import OrderedDict
from copy import deepcopy
from plotter import plotter

lines = get_lines()
colors = get_colors()

dir = "schan/"
hists = {}
vars = ["AEV","MT","MAOS"]
for var in vars:
    hists.update(import_attrs(dir+"bumphunt_{}/hists.py".format(var),"hists"))

components = [
    [OrderedDict([ # events
        ("QCD_AEV", {"counts": hists["AEV_15"]["counts"], "bins": hists["AEV_15"]["bins"], "color": colors[0], "linestyle": lines[0], "leg": "QCD"}),
        ("M1000_AEV", {"counts": hists["AEV_1000"]["counts"], "bins": hists["AEV_1000"]["bins"], "color": colors[1], "linestyle": lines[0], "leg": r"$m_{\mathrm{Z}^{\prime}} = 1000\,\mathrm{GeV}$"}),
        ("M2000_AEV", {"counts": hists["AEV_2000"]["counts"], "bins": hists["AEV_2000"]["bins"], "color": colors[2], "linestyle": lines[0], "leg": r"$m_{\mathrm{Z}^{\prime}} = 2000\,\mathrm{GeV}$"}),
        ("M3000_AEV", {"counts": hists["AEV_3000"]["counts"], "bins": hists["AEV_3000"]["bins"], "color": colors[3], "linestyle": lines[0], "leg": r"$m_{\mathrm{Z}^{\prime}} = 3000\,\mathrm{GeV}$"}),
        ("QCD_MT", {"counts": hists["MT_15"]["counts"], "bins": hists["MT_15"]["bins"], "color": colors[0], "linestyle": lines[1], "leg": ""}),
        ("M1000_MT", {"counts": hists["MT_1000"]["counts"], "bins": hists["MT_1000"]["bins"], "color": colors[1], "linestyle": lines[1], "leg": ""}),
        ("M2000_MT", {"counts": hists["MT_2000"]["counts"], "bins": hists["MT_2000"]["bins"], "color": colors[2], "linestyle": lines[1], "leg": ""}),
        ("M3000_MT", {"counts": hists["MT_3000"]["counts"], "bins": hists["MT_3000"]["bins"], "color": colors[3], "linestyle": lines[1], "leg": ""}),
        ("QCD_MAOS", {"counts": hists["MAOS_15"]["counts"], "bins": hists["MAOS_15"]["bins"], "color": colors[0], "linestyle": lines[2], "leg": ""}),
        ("M1000_MAOS", {"counts": hists["MAOS_1000"]["counts"], "bins": hists["MAOS_1000"]["bins"], "color": colors[1], "linestyle": lines[2], "leg": ""}),
        ("M2000_MAOS", {"counts": hists["MAOS_2000"]["counts"], "bins": hists["MAOS_2000"]["bins"], "color": colors[2], "linestyle": lines[2], "leg": ""}),
        ("M3000_MAOS", {"counts": hists["MAOS_3000"]["counts"], "bins": hists["MAOS_3000"]["bins"], "color": colors[3], "linestyle": lines[2], "leg": ""}),
        ("AEV", {"linestyle": lines[0], "color": "black", "leg": r"$V$"}),
        ("MT", {"linestyle": lines[1], "color": "black", "leg": r"$M_{\mathrm{T}}$"}),
        ("MAOS", {"linestyle": lines[2], "color": "black", "leg": r"$M_{\mathrm{MAOS}}$"}),
        ("y", {"name": r"events [138 $\mathrm{fb}^{-1}$]", "log": True}),
        ("x", {"range": [0, 6000]}),
        ("leg", {"on": True}),
    ])],
    [OrderedDict([ # signif
        ("y", {"name": r"$S/\sqrt{B}$", "log": False}),
    ])],
    [OrderedDict([ # signif_ratio
        ("line", {"counts": np.ones_like(hists["AEV_1000"]["counts"]), "bins": hists["AEV_1000"]["bins"], "color": "black", "linestyle": "dashed", "leg": ""}),
        ("x", {"name": "mass [calibrated]"}),
        ("y", {"name": r"$S/\sqrt{B}$ ratio ($V$/*)", "log": False, "range": [0, 2]}),
    ])],
]

for var in vars:
    for sig in ["M1000","M2000","M3000"]:
        svname = "{}_{}".format(sig,var)
        components[1][0][svname] = deepcopy(components[0][0][svname])
        numer = components[1][0][svname]["counts"]
        denom = np.sqrt(components[0][0]["QCD_{}".format(var)]["counts"])
        components[1][0][svname]["counts"] = np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0)

        if var=="AEV": continue
        components[2][0][svname] = deepcopy(components[0][0][svname])
        numer = components[1][0][svname.replace(var,"AEV")]["counts"]
        denom = components[1][0][svname]["counts"]
        components[2][0][svname]["counts"] = np.divide(numer, denom, out=np.zeros_like(numer), where=((denom!=0) & (np.asarray(components[0][0][svname]["bins"][:-1]) < (float(sig[1:])+800))))

plotter(components, [9], [7,2,2], dir, "signif_schan")
