from schan import config
from collections import OrderedDict

config.process = "schanNoMedMass"
config.params = OrderedDict([
    ("mZprime", [(500,1500),(3500,5000)]),
])
