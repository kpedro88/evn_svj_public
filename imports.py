import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import scipy.stats
import uproot as up
import awkward as ak
import vector as vec
import functools
vec.register_awkward()

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
import matplotlib.patheffects as path_effects