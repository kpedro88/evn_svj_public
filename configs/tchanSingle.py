from collections import OrderedDict
from magiconfig import MagiConfig

config = MagiConfig()
config.process = "tchanSingle"
config.filedir = "data/"
config.filenames = ["genvec_t-channel_nMed-1_mMed-{}_mDark-20_rinv-0.3_alpha-peak_yukawa-1_13TeV-madgraphMLM-pythia8_n-20000_part-1.root".format(mass) for mass in [500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]]
config.filenames_sep = OrderedDict([(mass, "genvec_t-channel_nMed-1_mMed-{}_mDark-20_rinv-0.3_alpha-peak_yukawa-1_13TeV-madgraphMLM-pythia8_n-20000_part-2.root".format(mass)) for mass in [500,1000,2000]])
config.xsecs_sep = OrderedDict([(500, 1.584e+01), (1000, 5.312e-01), (2000, 8.071e-03)])

config.params = OrderedDict([
    ("mMediator", (500,2000)),
])
config.inputs = [
    "Jet1",
    "Jet2",
    "Met",
]
config.theory = [
    "MT",
]

from axes_tchan import axes
config.axes = axes
