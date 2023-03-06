import numpy as np
from utils import import_attrs, get_lines, get_colors
from collections import OrderedDict
from copy import deepcopy
from plotter import plotter

lines = get_lines()
colors = get_colors()

rinvs = ["01", "03", "05", "07", "Mix"]
legnames = [
    r"r_{\mathrm{inv}} = 0.1",
    r"r_{\mathrm{inv}} = 0.3",
    r"r_{\mathrm{inv}} = 0.5",
    r"r_{\mathrm{inv}} = 0.7",
    r"\mathrm{mix~of~}r_{\mathrm{inv}}",
]

dirsTrain = [
    "schan01",
    "schan03",
    "schan05",
    "schan07",
    "schanMix",
]

components = [OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()]
hname = "AEV_2000"

for component,rinvTest,legnameTest in zip(components,rinvs[:-1],legnames[:-1]):
    for counter,(dirTrain,rinvTrain,legnameTrain) in enumerate(zip(dirsTrain,rinvs,legnames)):
        hists = import_attrs("{}/plotsRecalib{}/hists.py".format(dirTrain,rinvTest),"hists")
        component[hname+"_{}to{}".format(rinvTrain,rinvTest)] = {"counts": hists[hname]["counts"], "bins": hists[hname]["bins"], "color": colors[counter], "linestyle": lines[counter], "leg": r"$V_{"+legnameTrain+"}$"}
    component["rinv"] = {"leg": "$"+legnameTest+"$"}
    component["mZprime"] = {"leg": r"$m_{\mathrm{Z}^{\prime}} = 2000\,\mathrm{GeV}$"}
    component["y"] = {"name": r"arbitrary units", "log": False}
    component["x"] = {"name": r"mass [calibrated]", "range": [0, 4000]}
    component["leg"] = {"on": True}

# manual tweaks
components[0]["leg"] = {"on": True, "kwargs": {"loc": "upper left"}}
components[1]["leg"] = {"on": True, "kwargs": {"loc": "upper left"}}
components[0]["x"].pop("name")
components[1]["x"].pop("name")
components[1]["y"].pop("name")
components[3]["y"].pop("name")

# reshape
components = [
    [components[0], components[1]],
    [components[2], components[3]],
]

plotter(components, [9,9], [7,7], "schanMix", "rinv_comp")
