from dijetPair import config
from collections import OrderedDict

config.filenames = config.filenames[:-1]
config.params = OrderedDict([
    ("mStop", (500,900)),
])