from collections import OrderedDict
from magiconfig import MagiConfig

config = MagiConfig()
config.process = "qcdFlat"
config.filedir = "data/"
config.filenames = ["genvec_QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8_n-all.root"]
config.filenames_sep = OrderedDict([(15, "genvec_QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8_n-all.root")])
config.xsecs_sep = OrderedDict([(15, 2.0221e+09)])
config.weights = ["Weight"]
config.axes = {}
