# Reproducibility

This directory contains the materials needed to reproduce the experiments
reported in the paper "**Building and Breaking a Cross-Domain IoMT Network Intrusion Detection Systems**".

It includes [`dataset_sample.parquet`](dataset_sample.parquet), an example
Parquet file showing the dataset format expected by the framework. The full
datasets are not included in this repository. Follow the data-layout
instructions in the [main README](../README.md) and place the complete
datasets under the configured `base_data_path`.

## Semi-supervised ML approaches

```bash
./run_experiments.sh \
  --src-dataset cic_iomt,iomt \
  --trg-dataset cic_iomt,iomt \
  --approach label_propagation,label_spreading,st_random_forest \
  --seed 0-19 \
  --is-flat \
  --cpu 10
```

The launcher skips source/target pairs where both dataset names are equal.

## Anomaly-detection approaches

Run the classical anomaly detectors with flattened inputs:

```bash
./run_experiments.sh \
  --src-dataset cic_iomt,iomt \
  --trg-dataset cic_iomt,iomt \
  --approach isolation_forest,lof,ghmm,oc_svm \
  --seed 0-19 \
  --is-flat \
  --cpu 10
```

Run DAE3 with the additional options used for the experiments:

```bash
./run_experiments.sh \
  --src-dataset cic_iomt,iomt \
  --trg-dataset cic_iomt,iomt \
  --approach dae3 \
  --seed 0-19 \
  --is-flat \
  --cpu 10 \
  --extra-args "--gpu --n-thr 1"
```

## Unsupervised domain-adaptation approaches

```bash
./run_experiments.sh \
  --src-dataset cic_iomt,iomt \
  --trg-dataset cic_iomt,iomt \
  --approach cdan,mcc,adda,mcd,bsp,icon \
  --seed 0-19 \
  --cpu 10 \
  --extra-args "--max-epochs 200 --adapt-epochs 200 --n-thr 1 --n-tasks 2 --adapt-batch-size 64 --iter-per-epoch 1000 --discr-hidden-size 128 --gpu"
```

Run the baseline without target adaptation (``NoAdapt``):

```bash
./run_experiments.sh \
  --src-dataset cic_iomt,iomt \
  --trg-dataset cic_iomt,iomt \
  --approach baseline \
  --seed 0-19 \
  --cpu 10 \
  --extra-args "--max-epochs 200 --adapt-epochs 200 --n-thr 1 --n-tasks 2 --skip-t2"
```

## Adversarial attack experiments

After training a deep-learning approach, use its saved `dict_args.json` and
checkpoint file to run a constrained attack on the target test split. Run the
attack from `src/`:

```bash
CUDA_VISIBLE_DEVICES=0 python run_attack.py \
  --args-path <path/to/dict_args.json> \
  --ckpt-path <path/to/model_checkpoint.pt> \
  --output-fn <output-prefix> \
  --method <random|hsj|pgd|shap> \
  --query-budget 50 \
  --beta 0.25
```

The attack runner accepts only deep-learning experiment arguments and saves
the resulting DataFrame to `<output-prefix>.pkl`. The default feature mask is
`1 1 0 1 0 1` for `PL IAT DIR WIN FLG TTL`, leaving direction and flags
immutable. HopSkipJump-specific controls are available through
`--hsj-init-evals`, `--hsj-max-evals`, and `--hsj-gamma`.