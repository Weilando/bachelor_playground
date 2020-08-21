# Playground Bachelor Thesis
This framework provides experiments for my bachelor thesis about the _Lottery Ticket Hypothesis_.
It provides experiments with iterative magnitude pruning (IMP) and one-shot pruning (OSP).
These experiments are currently available:
- Finding winning tickets from fully-connected Lenet 300-100 on MNIST using IMP or OSP
- Finding winning tickets from convolutional Conv-2 and CIFAR-10 using IMP or OSP
- Finding winning tickets from convolutional Conv-4 and CIFAR-10 using IMP or OSP
- Finding winning tickets from convolutional Conv-6 and CIFAR-10 using IMP or OSP
- Retraining randomly reinitialized models from a previous IMP- or OSP-experiment

I developed and tested the framework using a conda-environment with packages specified in `conda_specs.txt`.
Use it to create a compatible environment.
If you want to use EC2 instances in AWS, I recommend the image "Deep Learning AMI (Amazon Linux 2)", as it offers AWS functionality (e.g. for S3) and configured conda-environments with Pytorch and CUDA out of the box. 

## Run experiments
There are two ways to perform experiments and both options allow execution on CPU or GPU (with CUDA).
- Firstly by executing the main module via `python -m playground <experiment>` from the root-directory, where `experiment` specifies the experiment to run.
Please use the flag `-h` to get more information on experiments and further flags and options.
Please use the flag `-l` to print the selected and modified specs without execution.
- Secondly by using the Jupyter notebook `~/experiments/ExperimentRunner.ipynb`.
It works with normal Jupyter backends and Google Colab.

There is a main experiment, which performs an IMP-experiment for several nets and levels of pruning.
Another experiment takes trained models from a previous IMP-experiment, applies new random weights and evaluates them during retraining.
To run the second experiment, it is necessary to generate checkpoints (an `EarlyStopHistoryList`) during the main experiment.

Implementations are in the package `experiments`.
For details on experiment specs have a look at `~/experiments/experiment_specs.py`).

### Predefined experiment specs for IMP and OSP

Experiment | Architecture | Dataset
--- | --- | ---
`lenet-mnist` | fc: 300-100-10 | MNIST
`conv2_cifar10` | conv: 64-64-M, fc: 256-256-10 | CIFAR-10
`conv4_cifar10` | conv: 64-64-M-128-128-M, fc: 256-256-10 | CIFAR-10
`conv6_cifar10` | conv: 64-64-M-128-128-M-256-256-M, fc: 256-256-10 | CIFAR-10

It is possible to specify other architectures by using the `--plan_conv` for convolutional layers and `--plan_fc` for fully-connected layers.
The `plan_fc`-option takes a list of numbers (either `int` or `str`) and interprets them as output-features of linear layers.
The networks automatically generate dimensions and inputs to achieve a working architecture.
It is not possible to alter the output-layer, i.e. it is always a fully-connected layer with ten output-features.
Additionally, the option `plan_conv` takes the options `A` for average-pooling, `M` for max-pooling and `iB` (with `i` integer) for a convolution followed by a batch-norm layer.

### Results
Experiments write their results into files in the subdirectory `~/data/results`, whereas `~/data/datasets` contains cached datasets.
The file-names start with the timestamp when the experiment finished, followed by the architecture and dataset.
The following suffixes specify the contents:

Suffix | Content | Previous format
--- | --- | ---
`-specs.json` | hyper-parameters and meta-data | `ExperimentSettings`
`-histories.npz` | histories of training-loss, validation-loss, validation-accuracy, test-accuracy and sparsity for several networks and pruning stages | `ExperimentHistories`
`-net<number>.pth` | state_dict of one net | `torch.nn.Module` (often subclasses like `Lenet` or `Conv`)
`-early-stop<number>.pth` | early-stop iterations and state_dicts of one net per level of pruning | `EarlyStopHistory`
`-random-histories<number>.npz` | histories of training-loss, validation-loss, validation-accuracy, test-accuracy and sparsity for several randomly reinitialized versions of one net for multiple pruning stages | `ExperimentHistories`

The dataclass `experiments.experiment_specs.ExperimentSpecs` contains all hyper-parameters like epoch count or initialization plans for networks.
Furthermore, it contains the absolute execution time and information about the used cuda-device.

The dataclass `experiments.experiment_histories.ExperimentHistories` contains an `np.array` per history and stores the measurements from one experiment.
All histories for training-loss, validation-loss, validation-accuracy and test-accuracy have the shape `(net_count, prune_count+1, data_length)`, which makes it easy to plot and evaluate them in arbitrary combinations.
The sparsity-history has shape `(prune_count+1)`.

The dataclass `experiments.early_stop_histories.EarlyStopHistoryList` contains an `np.array` with one `EarlyStopHistory` per net.
These store early-stop iterations and state_dicts per pruning level, if the `save_early_stop`-flag was set during training.

## Evaluate experiments
The module `experiments.result_loader` provides methods to load the stored results into their original data-structure and the module `data.plotter` brings high-level functions to analyze and plot histories in different contexts.
The interactive Jupyter notebook `ExperimentEvaluation.ipynb` shows examples for many use cases.

The high-level plot functions take a `matplotlib-axes` object and apply plots, labels and axes. 
Usually it is not necessary to call low-level functions by hand.
This design allows highly adaptive plots, as the user can arrange, combine and scale the provided plot types along others.
It is up to the user to look at the plots in a notebook or to save them to file for further processing. 
- `plot_average_hists_on_ax(...)` calculates the means of all networks for a given history and plots them per sparsity.
It plots the baseline, i.e. the run with the lowest sparsity, as dashed line and every further level of sparsity as solid line.
Iterations index the x-axis.
- `plot_acc_at_early_stop_on_ax(...)` applies the early-stopping criterion on a given loss and plots averages of a second history (typically an accuracy) per sparsity.
- `plot_early_stop_iterations_on_ax(...)` applies the early-stopping criterion on a given loss and plots averages of early-stopping iterations per sparsity.

Most high-level plot functions offer arguments of type `data.plotter.PlotType`.
This enum makes it easy to specify which kind of history a certain plot shows, and is used to generate suitable titles and labels.

All high-level plot functions take histories from randomly reinitialized nets as optional arguments.
Original plots have solid lines and random plots dotted lines, but corresponding lines (e.g. the same level of sparsity) have the same colors.
Baseline plots have dashed lines. 

It is possible to plot equal histories at early-stop for several networks.
Simply call the function with the same axes-object for each history to plot.
You should set the argument `setup_ax` to False until the last plot to have a complete legend and correct grids.

## Plot layers
The module `data.plotter_layer` provides high-level functions to plot linear and convolutional layers of trained networks.
The interactive Jupyter notebook `ExperimentEvaluation.ipynb` shows examples for these functions, too.

- `plot_conv(...)` plots all Conv2D-layers from a given Sequential.
- `plot_fc(...)` plots all Linear-layers from a given Sequential.

Rectangles with colored pixels represent weight matrices of linear layers.
Similarly, rows of rectangles represent kernels from convolutional layers.
If the convolutional layer has three input channels, one RGB plot shows a kernel, otherwise one plot exists per kernel and channel.
Both functions show masked weights as black dots.

Please keep in mind that convolutional plots might scale quickly, e.g. a network with two convolutions with 64 channels each produce a plot with 64 RGB images and a plot with 64*64=4096 single-channel plots.
`plot_conv(...)` shows at most 512 images.

## Tests
There are many unit and integration tests which cover correct experiment setup, execution and evaluation.
They also indicate if the Python environment provides all necessary packages.
Run them by calling `python -m unittest` from the main directory.
`~/test` contains all test files.

## IMP
After training and pruning the models, reset the survived weights to their initial values.
Then train and prune the subnetworks iteratively.

## OSP
After training the models once, prune them with steps of different sizes.
Always prune from the original network, then train and evaluate the subnetworks.
The pruning rates mimic levels of sparsity as generated by IMP-executions.