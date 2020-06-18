# Playground Bachelor Thesis
This repository contains experiments for my bachelor thesis about the _Lottery Ticket Hypothesis_.
Currently implemented experiments:
- Finding winning tickets from fully-connected Lenet 300-100 on MNIST using Iterative Magnitude Pruning
- Finding winning tickets from convolutional Conv-2 on CIFAR-10 using Iterative Magnitude Pruning
- Finding winning tickets from convolutional Conv-4 on CIFAR-10 using Iterative Magnitude Pruning
- Finding winning tickets from convolutional Conv-6 on CIFAR-10 using Iterative Magnitude Pruning

## Workflow
### Run experiments
One can find implemented experiments in the module `experiments`.
The experiments are subclasses of `Experiment` and can be executed by calling the main module via `python -m playground <experiment>` from the root-directory, where `experiment` specifies the experiment to run.
At the moment they differ in the used architectures (for details have a look at `experiment_settings.py`).

The following settings are available:

Experiment | Architecture | Dataset
--- | --- | ---
`lenet-mnist` | fc: 300-100-10 | MNIST
`conv2_cifar10` | conv: 64-64-M, fc: 256-256-10 | CIFAR-10
`conv4_cifar10` | conv: 64-64-M-128-128-M, fc: 256-256-10 | CIFAR-10
`conv6_cifar10` | conv: 64-64-M-128-128-M-256-256-M, fc: 256-256-10 | CIFAR-10

Please use the flag `-h` to get more information on further flags and options.

Experiments write their results into files in the subdirectory `/data/results`, whereas datasets are cached in `/data/datasets`.
The file-names start with the timestamp when the experiment finished, followed by the architecture and dataset.
The following suffixes specify the contents:

Suffix | Content | Previous format
--- | --- | ---
`-specs.json` | hyperparameters and meta-data | `dict`
`-histories.npz` | histories of loss, validation-accuracy, test-accuracy and sparsity for several networks and pruning stages | `np.array` (each)
`-net<number>.pth`| trained models (each stored in a single file) | `torch.nn.Module` (often subclasses like `Lenet` or `Conv`)

### Evaluate experiments
It is possible to load the stored results into their previous data-structure by using methods from the module `experiments.result_loader`.
The interactive Jupyter notebook `ExperimentEvaluation.ipynb` makes it easy to load and evaluate results from past experiments.
