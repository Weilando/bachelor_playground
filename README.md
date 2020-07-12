# Playground Bachelor Thesis
This framework provides experiments for my bachelor thesis about the _Lottery Ticket Hypothesis_.
These experiments are currently available:
- Finding winning tickets from fully-connected Lenet 300-100 on MNIST using Iterative Magnitude Pruning
- Finding winning tickets from convolutional Conv-2 on CIFAR-10 using Iterative Magnitude Pruning
- Finding winning tickets from convolutional Conv-4 on CIFAR-10 using Iterative Magnitude Pruning
- Finding winning tickets from convolutional Conv-6 on CIFAR-10 using Iterative Magnitude Pruning

## Workflow
### Run experiments
There are two ways to perform experiments.
- Firstly by executing the main module via `python -m playground <experiment>` from the root-directory, where `experiment` specifies the experiment to run.
Please use the flag `-h` to get more information on further flags and options.
- Secondly by using the Jupyter notebook `~/experiments/ExperimentRunner.ipynb`.
It works with normal Jupyter backends and Google Colab.

Both options allow execution on CPU or GPU (with CUDA).

One can find implemented experiments in the package `experiments`.
For details on experiment settings have a look at `~/experiments/experiment_settings.py`).

The following predefined experiment settings are available:

Experiment | Architecture | Dataset
--- | --- | ---
`lenet-mnist` | fc: 300-100-10 | MNIST
`conv2_cifar10` | conv: 64-64-M, fc: 256-256-10 | CIFAR-10
`conv4_cifar10` | conv: 64-64-M-128-128-M, fc: 256-256-10 | CIFAR-10
`conv6_cifar10` | conv: 64-64-M-128-128-M-256-256-M, fc: 256-256-10 | CIFAR-10

#### Results
Experiments write their results into files in the subdirectory `~/data/results`, whereas `~/data/datasets` contains cached datasets.
The file-names start with the timestamp when the experiment finished, followed by the architecture and dataset.
The following suffixes specify the contents:

Suffix | Content | Previous format
--- | --- | ---
`-specs.json` | hyper-parameters and meta-data | `ExperimentSettings`
`-histories.npz` | histories of training-loss, validation-loss, validation-accuracy, test-accuracy and sparsity for several networks and pruning stages | `ExperimentHistories`
`-net<number>.pth`| trained models (each stored in a single file) | `torch.nn.Module` (often subclasses like `Lenet` or `Conv`)

The dataclass `experiments.experiment_settings.ExperimentSettings` contains all hyper-parameters like epoch count or initialization plans for networks.
Furthermore, it contains the absolute execution time and information about the used cuda-device.

The dataclass `experiments.experiment_histories.ExperimentHistories` contains an `np.array` per history and stores the measurements from one experiment.
All histories for training-loss, validation-loss, validation-accuracy and test-accuracy have the shape `(net_count, prune_count+1, data_length)`, which makes it easy to plot and evaluate them in arbitrary combinations.
The sparsity-history has shape `(prune_count+1)`.

### Evaluate experiments
It is possible to load the stored results into their original data-structure by using methods from the module `experiments.result_loader`.
The interactive Jupyter notebook `ExperimentEvaluation.ipynb` makes it easy to load and evaluate results from past experiments.

The module `data.plotter` provides high-level functions to analyze and plot histories in different contexts.
In most cases there is no need to call low-level functions from the module directly.
- `plot_average_hists(...)` calculates the means of all networks for a given history and plots them per sparsity.
It plots the baseline, i.e. the run with the lowest sparsity, as dashed line and every further level of sparsity as solid line.
Iterations index the x-axis.
- `plot_two_average_hists(...)` works like the previous function, but plots two histories next to each other.
- `plot_acc_at_early_stop(...)` applies the early-stopping criterion on a given loss and plots averages of a second history (typically an accuracy) per sparsity.
- `plot_early_stop_iterations(...)` applies the early-stopping criterion on a given loss and plots averages of early-stopping iterations per sparsity.

All high-level plot functions offer arguments of type `data.plotter.PlotType`.
This enum makes it easy to specify which kind of history a certain plot shows, and is used to generate suitable titles and labels.

## Tests
There are many unit and integration tests which cover correct experiment setup, execution and evaluation.
Run them by calling `python -m unittest` from the main directory.
`~/test` contains all test files.