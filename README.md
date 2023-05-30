# Signatures for Paths of Persistence Diagrams

This repository contains the code to perform the computations in the paper "Signatures, Lipschitz-free spaces, and paths of persistence diagrams" by Chad Giusti and Darrick Lee.

This contains the following notebooks:

* `run_experiments.ipynb`: Run the experiments in Section 7.5 with the same train/test set.
* `run_heterogeneous_experiments.ipynb`: Run experiments in Section 7.5 with heterogeneous train/test sets.
* `compute_python_features.ipynb`: Notebook to compute persistence landscapes and images using the giotto-tda package in Python. (Used together with the two previous notebooks.)
* `kme_analysis.ipynb`: Run the sliced Wasserstein KME experiments in Section 7.6.
* `kme_timing_experiments.ipynb`: Short notebook which contains the timing experiments to compare KME runtime with signature runtime from Appendix D.