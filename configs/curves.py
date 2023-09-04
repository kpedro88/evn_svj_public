from collections import OrderedDict
curves = OrderedDict([
    ("AEV", {"leg": r"$V$", "folder": ["dijetPair_sep4_1","dijetPair_sep4_2","dijetPair_sep4_3"]}),
    ("AEV_low", {"leg": r"$V_{m_{\widetilde{\mathrm{t}}} < 1000\,\mathrm{GeV}}$", "folder": ["dijetPairLowMass_sep4_1","dijetPairLowMass_sep4_2","dijetPairLowMass_sep4_3"]}),
    ("AEV_high", {"leg": r"$V_{m_{\widetilde{\mathrm{t}}} > 500\,\mathrm{GeV}}$", "folder": ["dijetPairHighMass_sep4_1","dijetPairHighMass_sep4_2","dijetPairHighMass_sep4_3"]}),
])
