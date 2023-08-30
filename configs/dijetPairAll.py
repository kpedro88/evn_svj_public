from dijetPair import config
from collections import OrderedDict

config.filenames_sep = OrderedDict([(mass, "tree_ML_MCRun2_{}GeV.root".format(mass)) for mass in [500,600,700,800,900,1000]])
config.xsecs_sep = OrderedDict([(mass, 1) for mass in [500,600,700,800,900,1000]])
