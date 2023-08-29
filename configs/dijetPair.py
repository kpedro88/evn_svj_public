from collections import OrderedDict
from magiconfig import MagiConfig

config = MagiConfig()
config.process = "dijetPair"
config.filedir = "data/"
config.filenames = ["tree_ML_MCRun2_{}GeV.root".format(mass) for mass in [500,600,700,800,900,1000]]
config.filenames_sep = OrderedDict([(mass, "tree_ML_MCRun2_{}GeV.root".format(mass)) for mass in [500,700,1000]])
config.xsecs_sep = OrderedDict([(500, 1), (700, 1), (1000, 1)])

config.params = OrderedDict([
    ("mStop", (500,1000)),
])
config.inputs = [
    "Jet1",
    "Jet2",
    "Jet3",
    "Jet4",
]
config.theory = [
    "dR_M",
    "Truth_high_M",
    "Truth_avg_M",
]

from axes_dijet import axes
config.axes = axes
