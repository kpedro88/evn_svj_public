from collections import OrderedDict
curves = OrderedDict([
    ("AEV", {"leg": r"$V$", "folder": ["dijetPair_aug29_5","dijetPair_aug29_6","dijetPair_aug29_7","dijetPair_aug29_8"]}),
    ("AEV_low", {"leg": r"$V_{m_{\widetilde{\mathrm{t}}} > 500\,\mathrm{GeV}}$", "folder": ["dijetPairLowMass_sep1_1","dijetPairLowMass_sep1_2","dijetPairLowMass_sep1_3"]}),
    ("AEV_high", {"leg": r"$V_{m_{\widetilde{\mathrm{t}}} < 1000\,\mathrm{GeV}}$", "folder": ["dijetPairHighMass_sep1_1","dijetPairHighMass_sep1_2","dijetPairHighMass_sep1_3"]}),
])
