# Code for "Optimal Mass Variables for Semivisible Jets"

## Setup

Environment (every time):
```
singularity run -p --nv --bind /cvmfs/unpacked.cern.ch /cvmfs/unpacked.cern.ch/registry.hub.docker.com/fnallpc/fnallpc-docker:tensorflow-2.10.0-gpu-singularity
```

Installation (first time):
```
HOME=$PWD pip install --user --no-cache-dir 'magiconfig<3'
```

## Event generation

The script [svjHelper.py](./gen/svjHelper.py) can produce PYTHIA and MadGraph settings (if any) for different signal processes.
The options for this script can be viewed by invoking `python svjHelper.py --help`.

These settings should be combined with the common settings in [common.txt](./gen/common.txt) and the tune-specific settings in `tune_*.txt` and processed by the appropriate software versions in order to reproduce the data used in the paper.

The generation of QCD background events uses the settings in [flat_QCD.txt](./gen/flat_QCD.txt).

(The MadGraph template files, [DMsimp_SVJ_t](./gen/DMsimp_SVJ_t), originate from [smsharma/SemivisibleJets](https://github.com/smsharma/SemivisibleJets).)

## Running

The options for all scripts can be viewed by invoking `python [script].py --help`.

[`train.py`](./train.py) trains the composite network.

[`test.py`](./test.py) produces the 2D correlation plots and computes calibrations.

[`analyze.py`](./analyze.py) produces the 1D distribution plots.

[`plot_loss.py`](./plot_loss.py) plots losses from training.

[`bumphunt.py`](./bumphunt.py) produces the signal and background histograms for sensitivity studies.

[`plotter.py`](./plotter.py) and the various scripts that invoke it produce plots from serialized histograms from other scripts ([`analyze.py`](./analyze.py), [`bumphunt.py`](./bumphunt.py)).
