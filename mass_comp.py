import numpy as np
from utils import import_attrs, get_lines, get_colors
from collections import OrderedDict
from copy import deepcopy
from plotter import plotter

lines = get_lines()
colors = get_colors()

legnames = [
    r"$V$",
    r"$V_{m_{\widetilde{\mathrm{t}}} > 500\,\mathrm{GeV}}$",
    r"$V_{m_{\widetilde{\mathrm{t}}} < 1000\,\mathrm{GeV}}$",
]

dirs = [
    "dijetPair_aug29_8",
    "dijetPairLowMass_sep1_3",
    "dijetPairHighMass_sep1_3",
]
hnames = [
    "AEV_500",
    "AEV_700",
    "AEV_1000",
]
components = [OrderedDict(), OrderedDict(), OrderedDict()]

for counter,(component,hname) in enumerate(zip(components,hnames)):
    for counter2,(dir,legname) in enumerate(zip(dirs,legnames)):
        hists = import_attrs("{}/plotsPairAvg/hists.py".format(dir),"hists")
        component[hname+"_"+dir] = {"counts": hists[hname]["counts"], "bins": hists[hname]["bins"], "color": colors[counter2], "linestyle": lines[counter2], "leg": legname}
    component["mStop"] = {"leg": r"$m_{\widetilde{\mathrm{t}}} = "+str(hname.split('_')[1])+"\,\mathrm{GeV}$"}
    component["y"] = {"name": r"arbitrary units", "log": False}
    component["x"] = {"name": r"mass [calibrated]", "range": [250, 1250]}
    component["leg"] = {"on": True}

# manual tweaks
components[1]["y"].pop("name")
components[2]["y"].pop("name")

plotter([components], [9,9,9], [7], dirs[0], "mass_comp")
