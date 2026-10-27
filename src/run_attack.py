import json
import os
import sys
from argparse import ArgumentParser
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from approach.approach_factory import get_approach
from data.data_manager import DataManager
from adversarial_attacks.perturbation import Perturbation
from adversarial_attacks.perturbation_utils import DEFAULT_FEAT_MASK, FEAT_NAMES
from util.seed import seed_everything


METHODS = ['shap', 'random', 'hsj', 'pgd']


def aggregate_results(df):
    evaded = df['evaded']
    method = df['method'].iloc[0] if len(df) > 0 else 'unknown'

    print(f"\n=== {method.upper()} ===")
    print(f"  Evasion rate       : {evaded.mean():.2%}  ({evaded.sum()}/{len(df)})")

    q_ev = df.loc[evaded, 'n_queries']
    q_nev = df.loc[~evaded, 'n_queries']
    print(f"  Queries (evaded)   : {q_ev.mean():.1f}  ±{q_ev.std():.1f}")
    print(f"  Queries (failed)   : {q_nev.mean():.1f}  ±{q_nev.std():.1f}")

    l2_ev = df.loc[evaded, 'delta_l2']
    l2_nev = df.loc[~evaded, 'delta_l2']
    print(f"  ‖δ‖₂  (evaded)    : {l2_ev.mean():.4f}  ±{l2_ev.std():.4f}")
    print(f"  ‖δ‖₂  (failed)    : {l2_nev.mean():.4f}  ±{l2_nev.std():.4f}")

    feat_budgets = np.stack(df['delta_per_feat'].values)   # (N, F)
    mean_per_feat = feat_budgets.mean(axis=0)
    print("  Avg |δ| per feature:")
    for name, val in zip(FEAT_NAMES, mean_per_feat):
        print(f"    {name}: {val:.4f}")


def main():
    parser = ArgumentParser(conflict_handler='resolve', add_help=True)
    parser.add_argument('-a', '--args-path', type=str, required=True)
    parser.add_argument('-c', '--ckpt-path', type=str, required=True)
    parser.add_argument('-o', '--output-fn', type=str, default='perturb')
    parser.add_argument('-m', '--method', choices=METHODS, default='random')
    parser.add_argument('-b', '--beta', type=float, default=0.05)
    parser.add_argument('-q', '--query-budget', type=int, default=50)
    parser.add_argument(
        '--feat-mask', type=int, nargs=6,
        default=DEFAULT_FEAT_MASK, metavar=tuple(FEAT_NAMES),
        help='Per-feature mutability mask (1 = mutable, 0 = immutable).',
    )
    # HSJ-specific arguments
    parser.add_argument(
        '--hsj-init-evals', type=int, default=10,
        help='Initial gradient-estimation queries per HSJ iteration.',
    )
    parser.add_argument(
        '--hsj-max-evals', type=int, default=100,
        help='Maximum gradient-estimation queries per HSJ iteration.',
    )
    parser.add_argument(
        '--hsj-gamma', type=float, default=1.0,
        help='HSJ gamma: controls the binary-search convergence threshold theta.',
    )

    script_args = parser.parse_args()

    with open(script_args.args_path) as f:
        exp_args = SimpleNamespace(**json.load(f))

    if exp_args.appr_type == 'ml':
        raise ValueError('Adversarial attacks are compatible only with DL approaches.')

    exp_args.return_quintuple = True
    seed_everything(exp_args.seed)

    dm = DataManager(exp_args)
    dm.prepare_datasets()

    trg_datamodule = dm.get_datamodule(split='trg')
    trg_loader = trg_datamodule.get_test_data()
    train_xy = (trg_datamodule.train_x, trg_datamodule.train_y)

    approach = get_approach(
        approach_name=exp_args.approach,
        fs_task=exp_args.k is not None,
        **vars(exp_args),
    )
    approach.load_checkpoint(script_args.ckpt_path)

    print('=' * 100)
    print(f"Checkpoint          : {script_args.ckpt_path}")
    print(f"Method              : {script_args.method}")
    print(f"Beta                : {script_args.beta}")
    print(f"Query budget        : {script_args.query_budget}")
    print(f"Feature mask        : {dict(zip(FEAT_NAMES, script_args.feat_mask))}")
    if script_args.method == 'hsj':
        print(f"HSJ init evals      : {script_args.hsj_init_evals}")
        print(f"HSJ max evals       : {script_args.hsj_max_evals}")
        print(f"HSJ gamma           : {script_args.hsj_gamma}")

    hsj_args = {
        'hsj_init_evals': script_args.hsj_init_evals,
        'hsj_max_evals':  script_args.hsj_max_evals,
        'hsj_gamma':      script_args.hsj_gamma,
    }

    p = Perturbation(
        method=script_args.method,
        approach=approach,
        loader=trg_loader,
        train_xy=train_xy,
        beta=script_args.beta,
        query_budget=script_args.query_budget,
        feat_mask=script_args.feat_mask,
        hsj_args=hsj_args,
    )
    df = p.perturbate()

    aggregate_results(df)

    out_path = f'{script_args.output_fn}.pkl'
    df.to_pickle(out_path)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()