import warnings

import numpy as np
import torch

from .transform_utils import _SYM_PARAMS, _LF


FEAT_NAMES = ['PL', 'IAT', 'DIR', 'WIN', 'FLG', 'TTL']

# Which features are immutable by default (DIR and FLG).
DEFAULT_FEAT_MASK = [1, 1, 0, 1, 0, 1]

GLOBAL_BOUNDS = {
    'IAT': (1.0, 60_000_000.0),
    'WIN': (1.0, 65_535.0),
    'FLG': (2.0, 194.0),
    'TTL': (1.0, 255.0),
}

PL_MAX: float = 1500.0

PL_BOUNDS_BY_PROTO = {
    '0':   (76.0,  PL_MAX),
    '1':   (46.0,  PL_MAX),
    '17':  (46.0,  PL_MAX),
    '2':   (40.0,  PL_MAX),
    '6':   (46.0,  PL_MAX),
    '58':  (48.0,  PL_MAX),
    '146': (28.0,  PL_MAX),
}


def raw_bounds_by_proto(proto, dtype):
    """
    Return (raw_min, raw_max) arrays of shape (1, F) for the given protocol.

    The payload-length minimum depends on the IP protocol; all other bounds are
    protocol-independent.
    """
    proto = str(proto)
    pl_min, pl_max = PL_BOUNDS_BY_PROTO.get(proto, (28.0, PL_MAX))

    raw_min = np.array([
        pl_min,
        GLOBAL_BOUNDS['IAT'][0],
        0.0,                        # DIR
        GLOBAL_BOUNDS['WIN'][0],
        GLOBAL_BOUNDS['FLG'][0],
        GLOBAL_BOUNDS['TTL'][0],
    ], dtype=dtype).reshape(1, -1)

    raw_max = np.array([
        pl_max,
        GLOBAL_BOUNDS['IAT'][1],
        1.0,                        # DIR
        GLOBAL_BOUNDS['WIN'][1],
        GLOBAL_BOUNDS['FLG'][1],
        GLOBAL_BOUNDS['TTL'][1],
    ], dtype=dtype).reshape(1, -1)

    return raw_min, raw_max


def compute_alpha_per_feat(feat_mask, beta=0.1):
    """
    Compute a per-feature step size in the normalised domain.

    For each mutable feature the step corresponds to a shift of
    β × (v_lin − v_min) in raw space, mapped through the symlog derivative at
    v_lin (the 75th-percentile boundary between the linear and log regions).

    Features with ``feat_mask[i] == 0`` or with a two-parameter symlog config
    (no linear region) receive alpha = 0.

    Raises a warning when a mutable feature is silently frozen because its
    symlog parameter list has the wrong length.
    """
    alphas = []
    for p, mask in zip(_SYM_PARAMS, feat_mask):
        if not mask:
            alphas.append(0.0)
            continue
        if len(p) != 3:
            warnings.warn(
                f"Mutable feature has a {len(p)}-element symlog param (expected 3); "
                "alpha set to 0 — the feature will not be perturbed.",
                stacklevel=2,
            )
            alphas.append(0.0)
            continue
        v_min, v_lin, v_clip = p
        linear_range = v_lin - v_min
        alpha = (1 - _LF) * np.log1p(beta * linear_range) / np.log1p(v_clip - v_lin)
        alphas.append(alpha)
    return alphas


def get_random_background(train_xy, n_samples, device):
    """
    Draw a stratified random background set of *n_samples* from *train_xy*.

    Samples are balanced across classes; the remainder (n_samples % n_classes)
    is distributed to the first classes in sorted order.
    """
    train_x, train_y = train_xy
    classes, counts = np.unique(train_y, return_counts=True)
    spc = n_samples // len(classes)
    rem = n_samples % len(classes)

    if np.min(counts) < spc:
        raise ValueError(
            f"Not enough samples per class for background set "
            f"(min available={np.min(counts)}, need={spc})."
        )

    selected = []
    for i, c in enumerate(classes):
        idx = np.where(train_y == c)[0]
        selected.extend(
            np.random.choice(idx, size=spc + (1 if i < rem else 0), replace=False)
        )

    return (
        torch.from_numpy(train_x[selected]).to(device),
        torch.from_numpy(train_y[selected]).to(device),
    )


def query_label(x, appr):
    """
    Query the classifier and return the predicted class index (0 or 1).
    """
    with torch.no_grad():
        _, logits = appr._predict_step(
            x, torch.zeros(1, dtype=torch.long, device=x.device)
        )
        return torch.argmax(logits, dim=1).item()


def protocol_applicability_mask(x_orig, proto):
    """
    Return a (F,) float tensor with 1 for features that are applicable to
    *proto* and 0 for those that are not.

    Currently only TCP (proto 6) supports WIN and FLG; all other protocols
    have those two features zeroed out.
    """
    app = np.ones(len(FEAT_NAMES), dtype=np.float32)
    if str(proto) != '6':
        app[FEAT_NAMES.index('WIN')] = 0.0
        app[FEAT_NAMES.index('FLG')] = 0.0
    return torch.tensor(app, dtype=x_orig.dtype, device=x_orig.device)


def get_proto_from_quintuple(quintuple: str) -> str:
    """Extract the IP protocol number from a quintuple string."""
    return str(quintuple).split(',')[-1].split('_')[0]