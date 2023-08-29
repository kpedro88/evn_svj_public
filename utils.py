from imports import *

# set up path for configs
def config_path(basedir=""):
    import os,sys
    if len(basedir)==0: basedir = os.getcwd()
    sys.path.append(basedir+"/configs")

## Code block to print the Python and package versions
def print_versions():
    import platform
    print("Software versions")
    print(f"  * Python: {platform.python_version()}")
    print(f"    * numpy: {np.__version__}")
    print(f"    * scipy: {scipy.__version__}")
    print(f"    * matplotlib: {mpl.__version__}")
    print(f"    * tensorflow: {tf.__version__}")
    print(f"    * uproot: {up.__version__}")
    print(f"    * awkward: {ak.__version__}")
    print(f"    * vector: {vec.__version__}")

def set_random(seed):
    np.random.seed(seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    tf.random.set_seed(seed)
    return rng

def nth(odict,n):
    return list(odict.values())[n]

def first(odict):
    return nth(odict,0)

# Wrapper for plt.savefig ########################
def save_figure(pltobj, fname):
    _ = pltobj.savefig(fname, bbox_inches='tight')
    return _

def plot_format():
    # detect if in jupyter
    try:
        get_ipython
    except:
        mpl.use('Agg')

    plt.rcParams.update({'font.size': 20})

class ScalarFormatterForceFormat(ScalarFormatter):
    def __init__(self, fmt):
        self.__fmtstring = fmt
        super().__init__()
    def _set_format(self):
        self.format = self.__fmtstring

def get_lines():
    # the last one is dashdotdot
    return ["solid", "dashed", "dotted", "dashdot", (0, (3, 5, 1, 5, 1, 5))]

def get_colors():
    return ["#9c9ca1", "#e42536", "#5790fc", "#964a8b", "#f89c20", "#7a21dd", "#86c8dd"]

def get_markers():
    return ["o", "s", "d", "*", "v", "^", "<", ">", "P"]

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    attrsplit = attr if isinstance(attr,list) else attr.split('.')
    return functools.reduce(_getattr, [obj] + attrsplit)

def make_vector(arrays, qty, extras={}):
    qtys = {
        "pt": getattr(arrays,qty+"_pt"),
        "eta": rgetattr(arrays,qty+"_eta"),
        "phi": rgetattr(arrays,qty+"_phi"),
        "m": rgetattr(arrays,qty+"_m"),
    }
    for key,val in extras.items():
        qtys[key] = rgetattr(arrays,val)
    return ak.zip(qtys, with_name="Momentum4D")

def correl(x,y):
    kendalltau = scipy.stats.kendalltau(x, y, nan_policy='omit').correlation
    spearmanr = scipy.stats.spearmanr(x, y, nan_policy='omit').correlation

    flip = False
    if kendalltau < 0:
        flip = True
        kendalltau *= -1
        spearmanr *= -1

    return kendalltau, spearmanr, flip

def skewness(x,y,m):
    # get relative displacement from line y = mx:
    # y -> y/m
    # then rotate x,y by 45 degrees
    x = np.squeeze(x)
    y = np.squeeze(y)
    y = y/m
    xprime = (x+y)/np.sqrt(2.)
    yprime = (y-x)/np.sqrt(2.)
    rel = yprime/xprime
    skewness_val = scipy.stats.skew(rel, nan_policy='omit')
    return skewness_val

def calibrate(x, y, coeff=None, zero=True):
    from collections import OrderedDict
    import numpy as np

    degrees = [0,1]
    if zero or (coeff is not None and len(coeff)==1): degrees = [1]

    if "vals" in x or coeff is None:
        xvals = x["vals"]
        matrix = np.stack([xvals**d for d in degrees], axis=-1)
    if coeff is None:
        # nan check
        yvals = y["vals"]
        mask = ~np.isnan(xvals) & ~np.isnan(yvals)
        coeff = np.linalg.lstsq(matrix[mask], yvals[mask], rcond=None)[0]
    elif isinstance(coeff,list):
        coeff = np.array(coeff)
    # apply calibration to values
    if "vals" in x:
        fit = np.transpose(np.dot(matrix, coeff))
    else:
        fit = np.array([])
    if "vals_sep" in x:
        matrix_sep = OrderedDict([(x_key, np.stack([x_sep**d for d in degrees], axis=-1)) for x_key,x_sep in x["vals_sep"].items()])
        fit_sep = OrderedDict([(x_key, np.dot(x_sep, coeff)) for x_key, x_sep in matrix_sep.items()])
    else:
        fit_sep = OrderedDict()

    return coeff, fit, fit_sep

def import_attrs(pyname, attrs):
    if not isinstance(attrs,list):
        attrs = [attrs]
    tmp = __import__(pyname.replace(".py","").replace("/","."), fromlist=attrs)
    if len(attrs)==1:
        return getattr(tmp,attrs[0])
    else:
        return [getattr(tmp,attr) for attr in attrs]

def map_bool(L):
    return [bool(elem) for elem in L]
