import sys,os,shutil,glob
sys.path.append(".local/lib/python{}.{}/site-packages".format(sys.version_info.major,sys.version_info.minor))
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from argparse import _AppendAction, Action
from collections import OrderedDict
from data import EventData
from utils import config_path, import_attrs
config_path()

class OrderedDictAction(_AppendAction):
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None)
        if items is None:
            items = OrderedDict()
        else:
            items = self._copy_items(items)
        results = self.parse_params(values)
        items.update(results)
        setattr(namespace, self.dest, items)

    # from argparse
    def _copy_items(self, items):
        if items is None:
            return []
        # The copy module is used only in the 'append' and 'append_const'
        # actions, and it is needed only when the default value isn't a list.
        # Delay its import for speeding up the common case.
        if type(items) is list:
            return items[:]
        import copy
        return copy.copy(items)

    def parse_params(self, args):
        pass

class ParamAction(OrderedDictAction):
    def parse_params(self, args):
        results = []
        counter = 0
        while counter < len(args):
            name = str(args[counter])
            counter += 1

            pmin = float(args[counter])
            counter += 1

            pmax = float(args[counter])
            counter += 1

            results.append((name, (pmin,pmax)))
        return results

class FileSepAction(ParamAction):
    _final_type = str
    def parse_params(self, args):
        results = []
        counter = 0
        key = []
        for counter in range(len(args)):
            if counter==len(args)-1:
                results.append((tuple(key), str(args[counter])))
            else:
                try:
                    k = float(args[counter])
                except:
                    k = self._final_type(args[counter])
                key.append(k)
        return results

class XsecSepAction(FileSepAction):
    _final_type = float

class ImportAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values,str):
            item = import_attrs(values, self.dest)
            setattr(namespace, self.dest, item)

# wrapper to handle common arguments and checks
class EVNParser:
    def __init__(self, name, basic=False, model_default="models"):
        self.parser = ArgumentParser(config_options=MagiConfigOptions(), formatter_class=ArgumentDefaultsRawHelpFormatter)
        # common arguments
        self.parser.add_argument("--name", type=str, default=name, help="name of operation (used for saved config filename and output subdirectory)")
        self.parser.add_argument("-f", "--force", default=False, action="store_true", help="overwrite existing config")
        self.parser.add_argument("-v", "--verbose", default=False, action="store_true", help="enable verbose output")
        self.parser.add_argument("--outf", type=str, required=True, help="output folder")
        self.parser.add_argument("--model-dir", type=str, default=model_default, help="directory for saved model file")
        if basic: return
        self.parser.add_argument("--process", type=str, required=True, choices=sorted(list(EventData.processes)), help="process name")
        self.parser.add_argument("--filedir", type=str, required=True, help="directory for process files")
        self.parser.add_argument("--filenames", type=str, default=[], nargs='+', required=True, help="process files for training")
        self.parser.add_argument("--filenames-sep", metavar=("key1 [key2...] filename"), default=OrderedDict(), action=FileSepAction, nargs='*', help="separate process files for analysis (keys assumed to be float or str) (can be called multiple times)")
        self.parser.add_argument("--xsecs-sep", metavar=("key1 [key2...] xsec"), default=OrderedDict(), action=XsecSepAction, nargs='*', help="separate process cross sections for analysis (keys assumed to be float or str) (can be called multiple times)")
        self.parser.add_argument("--params", metavar=("name min max"), default=OrderedDict(), action=ParamAction, nargs='*', help="param (theta) name(s) & range(s) (can be called multiple times)")
        self.parser.add_argument("--inputs", type=str, default=[], nargs='+', help="input feature name(s)")
        self.parser.add_argument("--theory", type=str, default=[], nargs='+', help="theory variable name(s)")
        self.parser.add_argument("--extras", type=str, default=[], nargs='+', help="extra name(s)")
        self.parser.add_argument("--weights", type=str, default=[], nargs='+', help="weight name(s)")
        self.parser.add_argument("--selections", metavar=("name min max"), type=str, default=None, nargs='+', action=ParamAction, help="selection variable name(s) & range(s)")
        self.parser.add_argument("--axes", required=True, action=ImportAction, help="axis info file (provides dict of dicts)")

    def parse_args(self, argv=None, checker=None, save=lambda x: True):
        if argv is None: argv = sys.argv[1:]
        args = self.parser.parse_args(argv)
        # do any common error checking
        # do specific error checking
        if checker is not None:
            args = checker(args)
        # save the config
        if save(args):
            os.makedirs(args.outf, exist_ok=True)
            oname = "{}/config_{}.py".format(args.outf,args.name)
            if not args.force and os.path.isfile(oname):
                raise RuntimeError("Will not overwrite existing output config {}".format(oname))
            self.parser.write_config(args, oname)
        return args

    def get_process(self, args):
        process = EventData.processes[args.process](args.filedir, args.filenames, args.filenames_sep, args.params, args.inputs, args.theory, args.extras, args.weights, args.xsecs_sep, args.selections)
        return process
