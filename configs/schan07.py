from collections import OrderedDict
from magiconfig import MagiConfig

config = MagiConfig()
config.process = "schan07"
config.filedir = "data/"
config.filenames = ["genvec_SVJ_ScanZR_2018_TuneCP2_13TeV_pythia8_n-1.5e+06_part-1.root"]
config.filenames_sep = OrderedDict()
for mass in [1000,2000,3000]:
    for rinv in [0.7]:
        config.filenames_sep[mass] = "genvec_s-channel_mMed-{}_mDark-20_rinv-{}_alpha-peak_13TeV-pythia8_n-20000_part-1.root".format(mass,rinv)
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
