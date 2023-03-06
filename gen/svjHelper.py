import os, math, sys, shutil
from string import Template
from glob import glob
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

class quark(object):
    def __init__(self,id,mass):
        self.id = id
        self.mass = mass
        self.massrun = mass
        self.bf = 1
        self.on = True
        self.active = True # for running nf

    def __repr__(self):
        return str(self.id)+": m = "+str(self.mass)+", mr = "+str(self.massrun)+", on = "+str(self.on)+", bf = "+str(self.bf)

# follows Ellis, Stirling, Webber calculations
class massRunner(object):
    def __init__(self):
        # QCD scale in GeV
        self.Lambda = 0.218

    # RG terms, assuming nc = 3 (QCD)
    def c(self): return 1./math.pi
    def cp(self,nf): return (303.-10.*nf)/(72.*math.pi)
    def b(self,nf): return (33.-2.*nf)/(12.*math.pi)
    def bp(self,nf): return (153.-19.*nf)/(2.*math.pi*(33.-2.*nf))
    def alphaS(self,Q,nf): return 1./(self.b(nf)*math.log(Q**2/self.Lambda**2))

    # derived terms
    def cb(self,nf): return 12./(33.-2.*nf)
    def one_c_cp_bp_b(self,nf): return 1.+self.cb(nf)*(self.cp(nf)-self.bp(nf))

    # constant of normalization
    def mhat(self,mq,nfq):
        return mq/math.pow(self.alphaS(mq,nfq),self.cb(nfq))/self.one_c_cp_bp_b(nfq)

    # mass formula
    def m(self,mq,nfq,Q,nf):
        # temporary hack: exclude quarks w/ mq < Lambda
        alphaq = self.alphaS(mq,nfq)
        if alphaq < 0: return 0
        else: return self.mhat(mq,nfq)*math.pow(self.alphaS(Q,nf),self.cb(nf))*self.one_c_cp_bp_b(nf)

    # operation
    def run(self,quark,nfq,scale,nf):
        # run to specified scale and nf
        return self.m(quark.mass,nfq,scale,nf)

class quarklist(object):
    def __init__(self):
        # mass-ordered
        self.qlist = [
            quark(2,0.0023), # up
            quark(1,0.0048), # down
            quark(3,0.095),  # strange
            quark(4,1.275),  # charm
            quark(5,4.18),   # bottom
        ]
        self.scale = None
        self.runner = massRunner()

    def set(self,scale):
        self.scale = scale
        # mask quarks above scale
        for q in self.qlist:
            # for decays
            if scale is None or 2*q.mass < scale: q.on = True
            else: q.on = False
            # for nf running
            if scale is None or q.mass < scale: q.active = True
            else: q.active = False
        # compute running masses
        if scale is not None:
            qtmp = self.get(active=True)
            nf = len(qtmp)
            for iq,q in enumerate(qtmp):
                q.massrun = self.runner.run(q,iq,scale,nf)
        # or undo running
        else:
            for q in self.qlist:
                q.massrun = q.mass

    def reset(self):
        self.set(None)

    def get(self,active=False):
        return [q for q in self.qlist if (q.active if active else q.on)]

class svjHelper(object):
    def __init__(self):
        self.quarks = quarklist()
        self.alphaName = ""
        # parameters for lambda/alpha calculations
        self.n_c = 2
        self.n_f = 2
        self.b0 = 11.0/6.0*self.n_c - 2.0/6.0*self.n_f

    def setAlpha(self):
        self.alphaName = "peak"
        # "empirical" formula
        self.lambdaHV = 3.2*math.pow(self.mDark,0.8)
        self.alpha = self.calcAlpha(self.lambdaHV)

    # calculation of lambda to give desired alpha
    # see 1707.05326 fig2 for the equation: alpha = pi/(b * log(1 TeV / lambda)), b = 11/6*n_c - 2/6*n_f
    # n_c = HiddenValley:Ngauge, n_f = HiddenValley:nFlav
    # see also TimeShower.cc in Pythia8, PDG chapter 9 (Quantum chromodynamics), etc.

    def calcAlpha(self,lambdaHV):
        return math.pi/(self.b0*math.log(1000/lambdaHV))

    def calcLambda(self,alpha):
        return 1000*math.exp(-math.pi/(self.b0*alpha))

    def setModel(self,channel,mMediator,mDark,rinv,yukawa=None,nMediator=None):
        # store the basic parameters
        self.channel = channel
        self.mMediator = mMediator
        self.mDark = mDark
        self.rinv = rinv
        self.setAlpha()

        self.nMediator = None
        self.yukawa = None
        if self.channel=="t":
            self.nMediator = nMediator
            self.yukawa = yukawa

        # get more parameters
        self.mMin = self.mMediator-1
        self.mMax = self.mMediator+1
        self.mSqua = self.mDark/2. # dark scalar quark mass (also used for pTminFSR)

        # get limited set of quarks for decays (check mDark against quark masses, compute running)
        self.quarks.set(mDark)

    def getOutName(self):
        _outname = "SVJ"
        params = [
            ("channel", "{}-channel".format(self.channel)),
        ]
        if self.nMediator is not None: params.append(("nMediator", "nMed-{:g}".format(self.nMediator)))
        params.extend([
            ("mMediator", "mMed-{:g}".format(self.mMediator)),
            ("mDark", "mDark-{:g}".format(self.mDark)),
            ("rinv", "rinv-{:g}".format(self.rinv)),
            ("alpha", "alpha-{}".format(self.alphaName) if len(self.alphaName)>0 else "alpha-{:g}".format(self.alpha)),
        ])
        if self.yukawa is not None: params.append(("yukawa","yukawa-{:g}".format(self.yukawa)))
        for pname, pval in params:
            _outname += "_"+pval
        if self.channel=="s":
            _outname += "_13TeV-pythia8"
        else:
            _outname += "_13TeV-madgraphMLM-pythia8"

        return _outname

    def invisibleDecay(self,mesonID,dmID):
        lines = ['{:d}:oneChannel = 1 {:g} 0 {:d} -{:d}'.format(mesonID,self.rinv,dmID,dmID)]
        return lines

    def visibleDecay(self,type,mesonID,dmID):
        theQuarks = self.quarks.get()
        if type=="simple":
            # just pick down quarks
            theQuarks = [q for q in theQuarks if q.id==1]
            theQuarks[0].bf = (1.0-self.rinv)
        elif type=="democratic":
            bfQuarks = (1.0-self.rinv)/float(len(theQuarks))
            for iq,q in enumerate(theQuarks):
                theQuarks[iq].bf = bfQuarks
        elif type=="massInsertion":
            denom = sum([q.massrun**2 for q in theQuarks])
            # hack for really low masses
            if denom==0.: return self.visibleDecay("democratic",mesonID,dmID)
            for q in theQuarks:
                q.bf = (1.0-self.rinv)*(q.massrun**2)/denom
        else:
            raise ValueError("unknown visible decay type: "+type)
        lines = ['{:d}:addChannel = 1 {:g} 91 {:d} -{:d}'.format(mesonID,q.bf,q.id,q.id) for q in theQuarks if q.bf>0]
        return lines

    def getPythiaSettings(self):
        # todo: include safety/sanity checks

        lines_schan = [
            'HiddenValley:ffbar2Zv = on',
            # parameters for leptophobic Z'
            '4900023:m0 = {:g}'.format(self.mMediator),
            '4900023:mMin = {:g}'.format(self.mMin),
            '4900023:mMax = {:g}'.format(self.mMax),
            '4900023:mWidth = 0.01',
            '4900023:oneChannel = 1 0.982 102 4900101 -4900101',
            # SM quark couplings needed to produce Zprime from pp initial state
            '4900023:addChannel = 1 0.003 102 1 -1',
            '4900023:addChannel = 1 0.003 102 2 -2',
            '4900023:addChannel = 1 0.003 102 3 -3',
            '4900023:addChannel = 1 0.003 102 4 -4',
            '4900023:addChannel = 1 0.003 102 5 -5',
            '4900023:addChannel = 1 0.003 102 6 -6',
            # decouple
            '4900001:m0 = 50000',
            '4900002:m0 = 50000',
            '4900003:m0 = 50000',
            '4900004:m0 = 50000',
            '4900005:m0 = 50000',
            '4900006:m0 = 50000',
            '4900011:m0 = 50000',
            '4900012:m0 = 50000',
            '4900013:m0 = 50000',
            '4900014:m0 = 50000',
            '4900015:m0 = 50000',
            '4900016:m0 = 50000',
        ]

        # parameters for bifundamental mediators
        # (keep default flavor-diagonal couplings)
        bifunds = [4900001,4900002,4900003,4900004,4900005,4900006]
        lines_tchan = []
        for bifund in bifunds:
            lines_tchan.extend([
                '{:d}:m0 = {:g}'.format(bifund,self.mMediator),
                '{:d}:mMin = {:g}'.format(bifund,self.mMin),
                '{:d}:mMax = {:g}'.format(bifund,self.mMax),
                '{:d}:mWidth = 0.01'.format(bifund),
            ])
        lines_tchan.extend([
            # decouple
            '4900011:m0 = 50000',
            '4900012:m0 = 50000',
            '4900013:m0 = 50000',
            '4900014:m0 = 50000',
            '4900015:m0 = 50000',
            '4900016:m0 = 50000',
            '4900023:m0 = 50000',
        ])

        lines_decay = [
            # hidden spectrum:
            # fermionic dark quark,
            # diagonal pseudoscalar meson, off-diagonal pseudoscalar meson, DM stand-in particle,
            # diagonal vector meson, off-diagonal vector meson, DM stand-in particle
            '4900101:m0 = {:g}'.format(self.mSqua),
            '4900111:m0 = {:g}'.format(self.mDark),
            '4900211:m0 = {:g}'.format(self.mDark),
            '51:m0 = 0.0',
            '51:isResonance = false',
            '4900113:m0 = {:g}'.format(self.mDark),
            '4900213:m0 = {:g}'.format(self.mDark),
            '53:m0 = 0.0',
            '53:isResonance = false',
            # other HV params
            'HiddenValley:Ngauge = {:d}'.format(self.n_c),
            # when Fv has spin 0, qv spin fixed at 1/2
            'HiddenValley:spinFv = 0',
            'HiddenValley:FSR = on',
            'HiddenValley:fragment = on',
            'HiddenValley:alphaOrder = 1',
            'HiddenValley:Lambda = {:g}'.format(self.lambdaHV),
            'HiddenValley:nFlav = {:d}'.format(self.n_f),
            'HiddenValley:probVector = 0.75',
        ]
        # branching - effective rinv (applies to all meson species b/c n_f >= 2)
        # pseudoscalars have mass insertion decay, vectors have democratic decay
        lines_decay += self.invisibleDecay(4900111,51)
        lines_decay += self.visibleDecay("massInsertion",4900111,51)
        lines_decay += self.invisibleDecay(4900211,51)
        lines_decay += self.visibleDecay("massInsertion",4900211,51)
        lines_decay += self.invisibleDecay(4900113,53)
        lines_decay += self.visibleDecay("democratic",4900113,53)
        lines_decay += self.invisibleDecay(4900213,53)
        lines_decay += self.visibleDecay("democratic",4900213,53)

        lines = []
        if self.channel=="s": lines = lines_schan + lines_decay
        elif self.channel=="t": lines = lines_tchan + lines_decay

        return lines

    def getJetMatchSettings(self):
        lines = [
            'JetMatching:setMad = off', # if 'on', merging parameters are set according to LHE file
            'JetMatching:scheme = 1', # 1 = scheme inspired by Madgraph matching code
            'JetMatching:merge = on', # master switch to activate parton-jet matching. when off, all external events accepted
            'JetMatching:jetAlgorithm = 2', # 2 = SlowJet clustering
            'JetMatching:etaJetMax = 5.', # max eta of any jet
            'JetMatching:coneRadius = 1.0', # gives the jet R parameter
            'JetMatching:slowJetPower = 1', # -1 = anti-kT algo, 1 = kT algo. Only kT w/ SlowJet is supported for MadGraph-style matching
            'JetMatching:qCut = 125.', # this is the actual merging scale. should be roughly equal to xqcut in MadGraph
            'JetMatching:nJetMax = 2', # number of partons in born matrix element for highest multiplicity
            'JetMatching:doShowerKt = off', # off for MLM matching, turn on for shower-kT matching
        ]

        return lines

    def getMadGraphCards(self,base_dir,lhaid,events):
        if base_dir[-1]!='/': base_dir = base_dir+'/'

        # helper for templates
        def fill_template(inname, outname=None, **kwargs):
            if outname is None: outname = inname
            with open(inname,'r') as temp:
                old_lines = Template(temp.read())
                new_lines = old_lines.substitute(**kwargs)
            with open(inname,'w') as temp:
                temp.write(new_lines)
            if inname!=outname:
                shutil.move(inname,outname)

        mg_model_dir = os.path.expandvars(base_dir+"mg_model_templates")

        # replace parameters in relevant file
        param_args = dict(
            mediator_mass = "{:g}".format(self.mMediator),
            dark_quark_mass = "{:g}".format(self.mSqua),
        )
        if self.yukawa is not None: param_args["dark_yukawa"] = "{:g}".format(self.yukawa)
        fill_template(
            os.path.join(mg_model_dir,"parameters.py"),
            **param_args
        )

        # use parameters to generate card
        sys.path.append(mg_model_dir)
        from write_param_card import ParamCardWriter
        param_card_file = os.path.join(mg_model_dir,"param_card.dat")
        ParamCardWriter(param_card_file, generic=True)

        mg_input_dir = os.path.expandvars(base_dir+"mg_input_templates")
        modname = self.getOutName()
        template_paths = [p for ftype in ["dat","patch"] for p in glob(os.path.join(mg_input_dir, "*."+ftype))]
        for template in template_paths:
            fname_orig = os.path.join(mg_input_dir,template)
            fname_new = os.path.join(mg_input_dir,template.replace("modelname",modname))
            fill_template(
                fname_orig,
                fname_new,
                modelName = modname,
                totalEvents = "{:g}".format(events),
                lhaid = "{:g}".format(lhaid),
                # for t-channel
                procInclusive = "" if self.nMediator is None else "#",
                procPair = "" if self.nMediator==2 else "#",
                procSingle = "" if self.nMediator==1 else "#",
                procNonresonant = "" if self.nMediator==0 else "#",
            )

        return mg_model_dir, mg_input_dir

if __name__=="__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--channel", required=True, type=str, choices=["s","t"], help="production channel")
    parser.add_argument("--mMediator", required=True, type=float, help="mediator mass (GeV)")
    parser.add_argument("--mDark", required=True, type=float, help="dark hadron mass (GeV)")
    parser.add_argument("--rinv", required=True, type=float, help="invisible fraction")
    parser.add_argument("--yukawa", default=1.0, type=float, help="Yukawa coupling (for t-channel)")
    parser.add_argument("--nMediator", default=None, type=int, help="number of mediators (for t-channel exclusive production)")
    args = parser.parse_args()

    helper = svjHelper()
    helper.setModel(args.channel, args.mMediator, args.mDark, args.rinv, args.yukawa, args.nMediator)
    oname = helper.getOutName()

    pythia_lines = helper.getPythiaSettings()

    # extras for t-channel
    if helper.channel=="t":
        mg_name = "DMsimp_SVJ_t"
        lhaid = 325300
        nevents = 10000
        # copy template files
        data_path = os.path.expandvars("$PWD/"+mg_name)
        mg_dir = os.path.join(os.getcwd(),oname)
        # remove output directory if it already exists
        if os.path.isdir(mg_dir):
            shutil.rmtree(mg_dir)
        shutil.copytree(data_path,mg_dir)
        madgraph_dirs = helper.getMadGraphCards(mg_dir,lhaid,nevents)
        print("Madgraph card directories: {}".format(madgraph_dirs))

        matching_lines = helper.getJetMatchSettings()
        pythia_lines.extend(matching_lines)

    fname = "pythia_{}.txt".format(oname)
    with open(fname,'w') as file:
        file.write('\n'.join(pythia_lines))
    print("Pythia settings: {}".format(fname))

