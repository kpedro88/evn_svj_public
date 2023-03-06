from collections import OrderedDict
from magiconfig import MagiConfig

config = MagiConfig()
config.process = "schan"
config.filedir = "data"
config.filenames = ["genvec_SVJ_ScanZ_2018_TuneCP2_13TeV_pythia8_n-1e+06_part-1.root"]
config.filenames_sep = OrderedDict([(mass, "genvec_s-channel_mMed-{}_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-20000_part-1.root".format(mass)) for mass in [1000,2000,3000]])
config.xsecs_sep = OrderedDict([(1000, 10.95), (2000, 0.4537), (3000, 0.0412)])

config.params = OrderedDict([
    ("mZprime", (500,5000)),
])
config.inputs = [
    "Jet1",
    "Jet2",
    "Met",
]
config.theory = [
    "MT",
    "MAOS",
]

from axes import axes
config.axes = axes
