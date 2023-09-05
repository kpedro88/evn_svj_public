from collections import OrderedDict
curves = OrderedDict([
    ("AEV", {"leg": r"$V$", "folder": ["dijetPair_sep5_3","dijetPair_sep5_4"]}),
    ("AEV_low", {"leg": r"$V_{m_{\widetilde{\mathrm{t}}} < 1000\,\mathrm{GeV}}$", "folder": ["dijetPairLowMass_sep5_1","dijetPairLowMass_sep5_2"]}),
    ("AEV_high", {"leg": r"$V_{m_{\widetilde{\mathrm{t}}} > 500\,\mathrm{GeV}}$", "folder": ["dijetPairHighMass_sep5_1","dijetPairHighMass_sep5_2"]}),
])
