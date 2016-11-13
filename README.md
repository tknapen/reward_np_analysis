# reward_np_analysis
###analysis of multi-session reward project

this is the analysis repo of a set of fMRI experiments, in which the influence of reward on visual cortex is examined. Data are stored in a BIDS-compatible fashion, to be uploaded to openfMRI.org, and are preprocessed with a nipype workflow. 

Further analyses will focus BOLD time-course estimation, connectivity analyses between BOLD and pupil, etc.

## Prerequisites:

Python packages:
nipype, spynoza, fir, hedfpy, sklearn, numpy, scipy, matplotlib. 

Imaging software:
FSL, FreeSurfer, AFNI




# to do;

1. look up stream-up experiment's reward administration scheme - which index is which?
2. Add ocular regressors to whole-brain fMRI FIR and GLM analysis
3. Perform FIR analysis on pupil size signals:
- quality control of pupil size signals: throw out bad runs
- single-trial estimates of pupil response for pupil-BOLD regression (GLM?)
4. Get trial onset BOLD/pupil signal value, regress against each other.
5. Pearce-Hall model for BOLD/pupil trial-by-trial signals