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
        datatree = datafile["tree_ML"]
        # get unique, stable list of inputs
        qtys = list(self.params.keys())+self.inputs+self.theory+self.extras+self.weights+self.get_selvars()
        exprs = list(sorted(set([k for qty in qtys for k in self.get_qty_keys(qty)])))
        # handle requests for single component
        exprs_full = list(sorted(set([x.split('.')[0] for x in exprs])))
        arrays = datatree.arrays(expressions=exprs_full, library="ak")
        defs = {qty: self.def_event_qty(qty, datatree[qty.split('.')[0]].typename if qty.split('.')[0] in datatree else "", arrays, filename, datatree) for qty in qtys}
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
        if qty.startswith("Jet"):
            return [qty.replace("Jet","pt_ordered_jet")+suff for suff in ["_pt","_eta","_phi","_m"]]
        elif qty=="dR_M":
            return ["Mjj_avg_dRpairing_GeV"]
        elif "Truth" in qty:
            return ["Mjj_msortedP1_high","Mjj_msortedP1_low","Mjj_msortedP2_high","Mjj_msortedP2_low","Mjj_msortedP3_high","Mjj_msortedP3_low"]
        elif qty=="mStop":
            return [] # taken from filename
        else:
            return [qty]

    def def_event_qty(self, qty, typename, arrays, filename, tree):
        def get_mStop(filename):
            return float(filename.split('_')[-1].replace("GeV.root",""))
        keys = self.get_qty_keys(qty)
        if qty.startswith("Jet"):
            vec_tmp = make_vector(arrays,keys[-1][:-2])
            if '.' in qty:
                return getattr(vec_tmp, qty.split('.')[1])
            else:
                return vec_tmp
        elif qty=="mStop":
            return ak.from_numpy(get_mStop(filename)*np.ones(tree.num_entries))
        elif "Truth" in qty:
            mStop = get_mStop(filename)
            if "Truth_high" in qty:
                HM = np.array([
                    arrays["Mjj_msortedP1_high"],
                    arrays["Mjj_msortedP2_high"],
                    arrays["Mjj_msortedP3_high"]
                ])
                tmp = np.argmin(np.abs(HM - mStop),axis=0)
                if qty=="Truth_high":
                    return ak.from_numpy(tmp)
                elif qty=="Truth_high_M":
                    return ak.from_numpy(np.take_along_axis(HM,tmp[None],axis=0).T)
            elif "Truth_avg" in qty:
                AM = np.array([
                    (arrays["Mjj_msortedP1_high"]+arrays["Mjj_msortedP1_low"])/2.,
                    (arrays["Mjj_msortedP2_high"]+arrays["Mjj_msortedP2_low"])/2.,
                    (arrays["Mjj_msortedP3_high"]+arrays["Mjj_msortedP3_low"])/2.
                ])
                tmp = np.argmin(np.abs(AM - mStop),axis=0)
                if qty=="Truth_avg":
                    return ak.from_numpy(tmp)
                elif qty=="Truth_avg_M":
                    return ak.from_numpy(np.take_along_axis(AM,tmp[None],axis=0).T)
        # absent any specified calculation, use the key directly (renaming)
        elif len(keys)==1 and keys[0]!=qty:
            return arrays[keys[0]]
        else: # assume scalar
            return arrays[qty]

    def get_selvars(self):
        return []

    def selection(self, events):
        # cuts if requested
        if self.selections is not None:
            raise ValueError("Nothing implemented for selection(s): {}".format(self.selections))
        return events

    def _get_dims(self, category):
        dims = 0
        for key in getattr(self,category):
            if "Jet" in key:
                if '.' in key:
                    dims += 1
                else:
                    dims += 4
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
                if len(new_columns)>0:
                    columns.extend(new_columns)
                    continue

                qty = getattr(events,key)
                if "Jet" in key and '.' not in key:
                    columns.extend([np.asarray(qty.energy), np.asarray(qty.px), np.asarray(qty.py), np.asarray(qty.pz)])
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

class dijetPairData(EventData):
    pass
