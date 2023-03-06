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

## Running

The options for all scripts can be viewed by invoking `python [script]/py --help`.

`train.py` trains the composite network.

`test.py` produces the 2D correlation plots and computes calibrations.

`analyze.py` produces the 1D distribution plots.

`plot_loss.py` plots losses from training.

`bumphunt.py` produces the signal and background histograms for sensitivity studies.

`plotter.py` and the various scripts that invoke it produce plots from serialized histograms from other scripts (`analyze.py`, `bumphunt.py`).