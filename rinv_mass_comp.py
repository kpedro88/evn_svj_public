import numpy as np
from utils import import_attrs, get_lines, get_colors
from collections import OrderedDict
from copy import deepcopy
from plotter import plotter

lines = get_lines()
colors = get_colors()

rinvs = ["01", "03", "05", "07"]
legnames = [
    r"$r_{\mathrm{inv}} = 0.1$",
    r"$r_{\mathrm{inv}} = 0.3$",
    r"$r_{\mathrm{inv}} = 0.5$",
    r"$r_{\mathrm{inv}} = 0.7$",
]

dirTrain = "schanMix"

components = [OrderedDict(),OrderedDict()]
hnames = ["MAOS_2000","MT_2000"]
xnames = [r"$M_{\mathrm{MAOS}}$ [GeV]", r"$M_{\mathrm{T}}$ [GeV]"]

for component,hname,xname in zip(components,hnames,xnames):
    for counter,(rinv,legname) in enumerate(zip(rinvs,legnames)):
        hists = import_attrs("{}/plotsUncalib{}/hists.py".format(dirTrain,rinv),"hists")
        component[hname+"_{}".format(rinv)] = {"counts": hists[hname]["counts"], "bins": hists[hname]["bins"], "color": colors[counter], "linestyle": lines[counter], "leg": legname}
    component["mZprime"] = {"leg": r"$m_{\mathrm{Z}^{\prime}} = 2000\,\mathrm{GeV}$"}
    component["y"] = {"name": r"arbitrary units", "log": False}
    component["x"] = {"name": xname, "range": [0, 4000]}
    component["leg"] = {"on": True}

for component,hname in zip(components,hnames):
    plotter([[component]], [9], [7], dirTrain, "rinv_comp_{}".format(hname))
