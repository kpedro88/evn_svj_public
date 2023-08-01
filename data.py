from imports import *
from utils import *
from collections import OrderedDict

class EventData:
    processes = {}

    # register derived classes
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.processes[cls.__name__.replace("Data","")] = cls

    # params = OrderedDict, others = lists
    def __init__(self, filedir, filenames, filenames_sep, params, inputs, theory, extras, weights, xsecs_sep=None, selections=None, mask=None):
        # combine dir and filenames
        self.filenames = [filedir+("/" if filedir[-1]!="/" else "")+f for f in filenames]
        self.filenames_sep = OrderedDict([(k, filedir+("/" if filedir[-1]!="/" else "")+f) for k,f in filenames_sep.items()])

        # theta for EVN, w/ ranges/values
        self.params = params
        self.inputs = inputs
        self.theory = theory
        self.extras = extras
        self.weights = weights
        # to specify empty category on command line if already filled in config
        for category in ['inputs','theory','extras','weights']:
            if len(getattr(self,category))==1 and len(getattr(self,category)[0])==0:
                setattr(self,category,[])
        self.mask = mask
        self.selections = selections
        self.xsecs_sep = xsecs_sep

        for category in ["params","inputs","theory","extras","weights"]:
            self._get_dims(category)

    def get_events_file(self, filename):
        datafile = up.open(filename)
        datatree = datafile["GenVecAnalyzer/tree"]
        # get unique, stable list
        exprs = list(sorted(set([k for key in list(self.params.keys())+self.inputs+self.theory+self.extras+self.weights+self.get_selvars() for k in self.get_qty_keys(key)])))
        # handle requests for single component
        exprs_full = list(sorted(set([x.split('.')[0] for x in exprs])))
        arrays = datatree.arrays(expressions=exprs_full, library="ak")
        defs = {key: self.def_event_qty(key, datatree[key.split('.')[0]].typename, arrays) for key in exprs}
        # filter out possible None values (indicating def_event_qty() combined or discarded some exprs)
        defs = {key: val for key,val in defs.items() if val is not None}
        events_tmp = ak.zip(defs, depth_limit=1)
        return events_tmp

    def get_events_mask_sel(self, events):
        # apply user-provided mask (if any) before selection
        if self.mask is not None: events = events[np.squeeze(self.mask[""])]
        events = self.selection(events)
        return events

    def get_events(self):
        events = None
        for filename in self.filenames:
            events_tmp = self.get_events_file(filename)
            events = events_tmp if events is None else ak.concatenate([events,events_tmp],axis=0)
        events = self.get_events_mask_sel(events)
        return events

    def get_events_sep(self):
        events_sep = OrderedDict()
        for key,filename in self.filenames_sep.items():
            events_sep[key] = self.get_events_file(filename)
            events_sep[key] = self.get_events_mask_sel(events_sep[key])
        return events_sep

    def get_qty_keys(self, qty):
        if qty=="RT":
            return ["Met.pt","MT"]
        else:
            return [qty]

    def def_event_qty(self, key, typename, arrays):
        if "LorentzVector" in typename:
            vec_tmp = make_vector(arrays,key.split('.')[0])
            if '.' in key:
                return getattr(vec_tmp, key.split('.')[1])
            else:
                return vec_tmp
        # todo: add more cases
        elif "MAOS" in key:
            return np.clip(np.nan_to_num(arrays[key]),0,10000)
        else: # assume scalar
            return arrays[key]

    def get_selvars(self):
        return []

    def selection(self, events):
        # jet or met cuts if requested
        if self.selections is not None:
            if "jet1pt" in self.selections:
                cut = (events.Jet1.pt > self.selections["jet1pt"][0]) & (events.Jet1.pt < self.selections["jet1pt"][1])
            if "met" in self.selections:
                cut = (events.Met.pt > self.selections["met"][0]) & (events.Met.pt < self.selections["met"][1])
            events = events[cut]
        return events

    def _get_dims(self, category):
        dims = 0
        for key in getattr(self,category):
            if "Jet" in key:
                if '.' in key:
                    dims += 1
                else:
                    dims += 4
            elif "Met" in key:
                if '.' in key:
                    dims += 1
                else:
                    dims += 2
            else:
                dims += 1
        setattr(self,category+"_dim",dims)

    def _get_columns(self, events, category, event_list=None):
        if len(getattr(self,category))==0: return None

        columns_dict = OrderedDict()
        if isinstance(events,dict):
            events_dict = events
            return_dict = True
        else:
            events_dict = OrderedDict([
                ("", events)
            ])
            return_dict = False
        for dataset,events in events_dict.items():
            columns = []
            for key in getattr(self,category):
                new_columns = []
                if key=="RT":
                    new_columns.append(np.asarray(getattr(events,"Met.pt")/getattr(events,"MT")))
                if len(new_columns)>0:
                    columns.extend(new_columns)
                    continue

                qty = getattr(events,key)
                if "Jet" in key and '.' not in key:
                    columns.extend([np.asarray(qty.energy), np.asarray(qty.px), np.asarray(qty.py), np.asarray(qty.pz)])
                elif "Met" in key and '.' not in key:
                    columns.extend([np.asarray(qty.px), np.asarray(qty.py)])
                else:
                    columns.extend([np.asarray(qty)])
            columns = np.column_stack(columns)
            if event_list is not None and str(dataset) in event_list: columns = columns[np.squeeze(event_list[str(dataset)])]
            columns_dict[dataset] = columns
        if return_dict:
            return columns_dict
        else:
            return first(columns_dict)

    def get_inputs(self, events, event_list=None):
        return self._get_columns(events,"inputs",event_list)

    def get_params(self, events, event_list=None):
        return self._get_columns(events,"params",event_list)

    def get_theory(self, events, event_list=None):
        return self._get_columns(events,"theory",event_list)

    def get_extras(self, events, event_list=None):
        return self._get_columns(events,"extras",event_list)

    def get_weights(self, events, event_list=None):
        return self._get_columns(events,"weights",event_list)

class schanData(EventData):
    pass

class schanLowMassData(EventData):
    def get_selvars(self):
        selvars = [
            "mZprime",
        ]
        return selvars

    def selection(self, events):
        events = super().selection(events)
        cut = events.mZprime < 2501
        events = events[cut]
        return events

class schanHighMassData(EventData):
    def get_selvars(self):
        selvars = [
            "mZprime",
        ]
        return selvars

    def selection(self, events):
        events = super().selection(events)
        cut = events.mZprime > 2500
        events = events[cut]
        return events

class schanNoMedMassData(EventData):
    def get_selvars(self):
        selvars = [
            "mZprime",
        ]
        return selvars

    def selection(self, events):
        events = super().selection(events)
        cut = (events.mZprime < 1501) | (events.mZprime > 3500)
        events = events[cut]
        return events

class schanMixData(EventData):
    pass

class schan01Data(EventData):
    def get_selvars(self):
        selvars = [
            "rinv",
        ]
        return selvars

    def selection(self, events):
        events = super().selection(events)
        cut = np.abs(events.rinv - 0.1) < 0.001
        events = events[cut]
        return events

class schan03Data(EventData):
    def get_selvars(self):
        selvars = [
            "rinv",
        ]
        return selvars

    def selection(self, events):
        events = super().selection(events)
        cut = np.abs(events.rinv - 0.3) < 0.001
        events = events[cut]
        return events

class schan05Data(EventData):
    def get_selvars(self):
        selvars = [
            "rinv",
        ]
        return selvars

    def selection(self, events):
        events = super().selection(events)
        cut = np.abs(events.rinv - 0.5) < 0.001
        events = events[cut]
        return events

class schan07Data(EventData):
    def get_selvars(self):
        selvars = [
            "rinv",
        ]
        return selvars

    def selection(self, events):
        events = super().selection(events)
        cut = np.abs(events.rinv - 0.7) < 0.001
        events = events[cut]
        return events

class tchanSingleData(EventData):
    pass

class tchanPairData(EventData):
    pass

class qcdFlatData(EventData):
    pass
