import numpy as np
from utils import import_attrs, get_lines, get_colors
from collections import OrderedDict
from copy import deepcopy
from plotter import plotter

lines = get_lines()
colors = get_colors()

legnames = [
    r"$V_{m_{\mathrm{Z}^{\prime}} > 2500\,\mathrm{GeV}}$",
    r"$V_{m_{\mathrm{Z}^{\prime}} \leq 1500~\mathrm{or}~m_{\mathrm{Z}^{\prime}} > 3500\,\mathrm{GeV}}$",
    r"$V_{m_{\mathrm{Z}^{\prime}} \leq 2500\,\mathrm{GeV}}$",
]

dirBaseline = "schan"
histsBaseline = import_attrs("{}/plots/hists.py".format(dirBaseline),"hists")
dirsTrain = [
    "schanHighMass",
    "schanNoMedMass",
    "schanLowMass",
]
hnames = [
    "AEV_1000",
    "AEV_2000",
    "AEV_3000",
]
components = [OrderedDict(), OrderedDict(), OrderedDict()]

for counter,(component,dirTrain,legname,hname) in enumerate(zip(components,dirsTrain,legnames,hnames)):
    hists = import_attrs("{}/plotsRecalib/hists.py".format(dirTrain),"hists")
    # full training for comparison
    component[hname] = {"counts": histsBaseline[hname]["counts"], "bins": histsBaseline[hname]["bins"], "color": colors[0], "linestyle": lines[0], "leg": r"$V$"}
    component[hname+"_"+dirTrain] = {"counts": hists[hname]["counts"], "bins": hists[hname]["bins"], "color": colors[counter+1], "linestyle": lines[counter+1], "leg": legname}
    component["mZprime"] = {"leg": r"$m_{\mathrm{Z}^{\prime}} = "+str(hname.split('_')[1])+"\,\mathrm{GeV}$"}
    component["y"] = {"name": r"arbitrary units", "log": False}
    component["x"] = {"name": r"mass [calibrated]", "range": [0, 4000]}
    component["leg"] = {"on": True}

# manual tweaks
components[1]["y"].pop("name")
components[2]["y"].pop("name")

plotter([components], [9,9,9], [7], dirBaseline, "mass_comp")
