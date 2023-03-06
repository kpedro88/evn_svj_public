import numpy as np
from utils import import_attrs, get_lines, get_colors
from collections import OrderedDict
from copy import deepcopy
from plotter import plotter

lines = get_lines()
colors = get_colors()

jets = ["Jet1.pt","Jet2.pt","Jet3.pt"]
legnames = [
    r"$J_{1}$",
    r"$J_{2}$",
    r"$J_{3}$",
]

dirTrain = "tchanSingle"

component = OrderedDict()
mname = "1000"
xname = r"$p_{\mathrm{T}}$ [GeV]"
hists = import_attrs("{}/plotsPt/hists.py".format(dirTrain),"hists")

for counter,(jet,legname) in enumerate(zip(jets,legnames)):
    hname = "{}_{}".format(jet,mname)
    component[hname] = {"counts": hists[hname]["counts"], "bins": hists[hname]["bins"], "color": colors[counter], "linestyle": lines[counter], "leg": legname}
component["mMediator"] = {"leg": r"$m_{\Phi} = "+mname+r"\,\mathrm{GeV}$"}
component["y"] = {"name": r"arbitrary units", "log": True}
component["x"] = {"name": xname, "range": [0, 2000]}
component["leg"] = {"on": True}

plotter([[component]], [9], [7], dirTrain, "pt_comp_{}".format(mname))
