# Cross-Domain Network Intrusion Detection

This repository contains the code accompanying the paper "**Building and Breaking a Cross-Domain IoMT Network Intrusion Detection Systems**".

It provides an experimental framework for evaluating network intrusion detection across source and target datasets.
trains selected machine-learning or deep-learning approaches, 
evaluates source and target tasks, and saves metrics and plots to timestamped results directories.

## Features

- Packet-sequence input using packet length (`PL`), inter-arrival time (`IAT`),
  direction (`DIR`), TCP window (`WIN`), flags (`FLG`), and TTL (`TTL`).
- Min-max or symmetric-log preprocessing configured in `config.yaml`.
- Source-only training, source/target combined training, sequential
  source-to-target training, and unsupervised adaptation workflows.
- Classical ML, anomaly-detection, and neural domain-adaptation approaches.
- Constrained adversarial attacks against trained deep-learning models.
- Accuracy, macro/micro precision, recall and F1 metrics, classification
  reports, confusion matrices, and per-epoch plots.

## Requirements

Install the Python dependencies listed in [`requirements.txt`](requirements.txt)
in a Python environment. The project does not declare a Python version; use a
Python version supported by the installed versions of PyTorch, scikit-learn,
and the other dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
It is also recommended to install [GNU parallel](https://www.gnu.org/software/parallel/).

## Data layout

The built-in dataset registry in
[`src/data/dataset_config.py`](src/data/dataset_config.py) supports these
dataset names:

- `iomt`
- `cic_iomt`

With the default `base_data_path` in [`config.yaml`](config.yaml), the
repository expects the following files relative to the `src` working
directory:

```text
../../data/
├── iomt/
│   └── iomt_traffic.parquet
└── cic_iomt/
    └── cic-iomt.parquet
```

Each Parquet file must contain the configured label column (`LABEL-bin` for
both built-in datasets), the packet feature columns, and `FEAT_PAD`. The
reader creates cached preprocessed Parquet files and label-conversion JSON
files beside the source data.

To use other datasets, add an entry to `dataset_config` with its Parquet path
and label column, then pass the new registry key to `--src-dataset` or
`--trg-dataset`.

## Configuration

[`config.yaml`](config.yaml) contains defaults for:

- dataset paths, packet count, selected fields, padding, scaling, and splits;
- random seeds, device and thread settings, task mode, and logging;
- optimizer, scheduler, early-stopping, checkpoint, and dataloader settings;
- network and approach-specific hyperparameters.

The command-line parser loads this file using the relative path
`../config.yaml`, so commands should be run from `src/` unless the
configuration-loading code is changed. Command-line options override the
corresponding defaults.

Important data and experiment options include:

| Option | Description |
| --- | --- |
| `--src-dataset` | Source dataset registry key; required by the parser. |
| `--trg-dataset` | Optional target dataset registry key. |
| `--approach` | Approach name from the supported lists below. |
| `--n-tasks {1,2}` | Combined training (`1`) or sequential source/target tasks (`2`). |
| `--seed` | Reproducibility seed. |
| `--gpu` | Use CUDA when available. |
| `--is-flat` | Flatten packet-sequence input for ML approaches. |
| `--num-pkts` | Number of packets used from each sequence. |
| `--fields FIELD [...]` | Packet fields used as input. |
| `--log-dir` | Base directory for experiment output. |
| `--ckpt-path` | Checkpoint to load for a subsequent task. |

Use `python main.py --help` from `src/` for the complete parser-generated
option list, including approach-specific hyperparameters.

## Supported approaches

Machine-learning approaches:

```text
label_propagation
label_spreading
st_random_forest
isolation_forest
oc_svm
lof
ghmm
dae3
```

Deep-learning approaches:

```text
baseline
adda
mcc
cdan
mcd
bsp
icon
```

`label_propagation`, `label_spreading`, and `st_random_forest` are the
registered semi-supervised ML approaches. `adda`, `mcc`, `cdan`, `mcd`, `bsp`,
and `icon` are the registered unsupervised deep-learning approaches. The
anomaly-detection workflows are provided by `isolation_forest`, `oc_svm`,
`lof`, `ghmm`, and `dae3`.

## Running an experiment

Run the entry point from `src/`:

```bash
cd src
python main.py \
  --src-dataset iomt \
  --trg-dataset cic_iomt \
  --approach baseline \
  --seed 0 \
  --n-tasks 1
```

For source-only training, omit `--trg-dataset`:

```bash
cd src
python main.py \
  --src-dataset iomt \
  --approach oc_svm \
  --seed 0 \
  --is-flat
```

For sequential source-to-target training, use two tasks. The `--skip-t1` and
`--skip-t2` flags can skip the corresponding task:

```bash
cd src
python main.py \
  --src-dataset iomt \
  --trg-dataset cic_iomt \
  --approach adda \
  --n-tasks 2 \
  --seed 0
```

## Running experiment batches

[`src/run_experiments.sh`](src/run_experiments.sh) generates every
source/target/approach/seed combination and runs them in parallel. It requires
[GNU parallel](https://www.gnu.org/software/parallel/) for progress reporting; if [GNU parallel](https://www.gnu.org/software/parallel/) is unavailable, the
script uses `xargs`.

From `src/`, for example:

```bash
./run_experiments.sh \
  --src-dataset iomt \
  --trg-dataset cic_iomt \
  --seed 0-2 \
  --approach baseline,adda \
  --cpu 2
```

Optional arguments include `--is-flat`, `--log-keyword`, and
`--extra-args "<arguments passed to python main.py>"`.

## Adversarial attacks

The [`adversarial_attacks/`](adversarial_attacks/) package implements
constrained evasion attacks for trained deep-learning approaches. The attack
runner is [`src/run_attack.py`](src/run_attack.py). It loads the arguments
saved in an experiment's `dict_args.json`, restores a model checkpoint, and
attacks the target test split.

Attacks are currently supported only for deep-learning approaches. The
available methods are:

| Method | Description |
| --- | --- |
| `random` | Random constrained perturbations. |
| `hsj` | HopSkipJump decision-based black-box attack. |
| `pgd` | Projected Gradient Descent white-box attack. |
| `shap` | SHAP `GradientExplainer`-guided white-box attack. |

Run the command from `src/`:

```bash
CUDA_VISIBLE_DEVICES=0 python run_attack.py \
  --args-path <path/to/dict_args.json> \
  --ckpt-path <path/to/model_checkpoint.pt> \
  --output-fn <output-prefix> \
  --method hsj \
  --query-budget 50 \
  --beta 0.25
```

`--output-fn` is a filename prefix; the results are saved as
`<output-prefix>.pkl`. The output is a pandas DataFrame containing the
original and adversarial inputs in normalized and raw feature space, deltas,
per-feature perturbation magnitudes, the L2 perturbation norms, query counts,
the attack method, and the evasion result. Summary statistics are printed to
the console.

By default, the six-feature mutability mask is `1 1 0 1 0 1` for
`PL IAT DIR WIN FLG TTL`, respectively. This leaves direction and flags
immutable. Override it with six integer values:

```bash
python run_attack.py \
  --args-path <path/to/dict_args.json> \
  --ckpt-path <path/to/model_checkpoint.pt> \
  --method random \
  --feat-mask 1 1 0 1 0 1 \
  --query-budget 50
```

Additional HopSkipJump options are `--hsj-init-evals`, `--hsj-max-evals`, and
`--hsj-gamma`. The attack code also enforces packet-direction, padding,
protocol applicability, feature-rounding, and raw-domain bounds during
projection.

## Outputs

The default base output directory is `../results`. Each run creates a
timestamped directory containing a `src/` task directory and, when applicable,
a `trg/` task directory. Depending on the workflow, outputs include:

- `dict_args.json` with the resolved run arguments;
- phase outputs such as `labels.npz`, `preds.npz`, and result JSON files;
- training logs, checkpoints, and timing information;
- `detailed/report_<phase>.txt` metric reports;
- `detailed/confusion_matrix_<phase>.csv`;
- normalized confusion-matrix PDFs;
- `epoch_loss.pdf` and `epoch_metrics.pdf` when per-epoch logs are available.

Generated result files such as logs, checkpoints, CSVs, PDFs, and NumPy
archives are ignored by Git.

## Project structure

```text
src/
├── main.py                 # Command-line entry point
├── run_experiments.sh      # Batch experiment launcher
├── approach/               # ML and deep-learning approaches
├── data/                   # Dataset registry, reader, preprocessing, splits
├── module/                 # Neural modules and losses
├── network/                # Network implementations and factory
├── trainer/                # Training, adaptation, and evaluation workflows
├── callback/               # Checkpointing, logging, outputs, and timing
└── util/                   # Argument parsing, configuration, seeding, logging
```

The repository root also contains `adversarial_attacks/` and the
[`reproducibility/`](reproducibility/) materials used to reproduce the paper
experiments.

## License

This project is distributed under the MIT License. See
[`LICENSE`](LICENSE) for the full text.

## Acknowledgement

We thank the following open-source implementations that were used in this work:

- [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library)
- [ICON](https://github.com/yue-zhongqi/ICON)
- [GNU parallel](https://www.gnu.org/software/parallel/)

## Citation
If you use this framework, please cite:
```
@article{ref,
  title={Building and Breaking a Cross-Domain IoMT Network Intrusion Detection Systems},
  author={},
  journal={},
  year={},
  publisher={},
  doi={}
}
```
