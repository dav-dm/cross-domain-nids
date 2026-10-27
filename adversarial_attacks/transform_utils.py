import numpy as np
import torch
from util.config import load_config


#---------------------------------------------------------------------------
# Symlog transform
#---------------------------------------------------------------------------

_cf = load_config()
_SYM_PARAMS = [_cf['symlog_params'][f] for f in ('PL', 'IAT', 'DIR', 'WIN', 'FLG', 'TTL')]
_LF = _cf['linear_fraction']

def _symlog(x, p, lf=_LF):
    """Map raw feature values to the normalised [0, 1] domain via symlog."""
    x = np.asarray(x, dtype=np.float64)
    if len(p) == 2:
        v_min, v_clip = p
        return (np.clip(x, v_min, v_clip) - v_min) / (v_clip - v_min)
    v_min, v_lin, v_clip = p
    x = np.clip(x, v_min, v_clip)
    out = np.empty_like(x)
    m = x <= v_lin
    out[m] = (x[m] - v_min) / (v_lin - v_min) * lf
    out[~m] = lf + (1 - lf) * np.log1p(x[~m] - v_lin) / np.log1p(v_clip - v_lin)
    return out


def _inv_symlog(x_norm: np.ndarray, p: list, lf: float = _LF) -> np.ndarray:
    """Invert the symlog transform: normalised [0, 1] → raw feature values."""
    x_norm = np.clip(np.asarray(x_norm, dtype=np.float64), 0.0, 1.0)
    if len(p) == 2:
        v_min, v_clip = p
        return x_norm * (v_clip - v_min) + v_min
    v_min, v_lin, v_clip = p
    out = np.empty_like(x_norm)
    m = x_norm <= lf
    out[m] = x_norm[m] * (v_lin - v_min) / lf + v_min
    if (~m).any():
        ratio = (x_norm[~m] - lf) / (1.0 - lf)
        out[~m] = v_lin + np.expm1(ratio * np.log1p(v_clip - v_lin))
    return out

#---------------------------------------------------------------------------
# Feature rounding
#---------------------------------------------------------------------------

# Which features are integers in raw space (PL, IAT, DIR, WIN, FLG, TTL).
FEAT_INTEGER = [True, False, False, True, False, True]

def from_norm_to_raw(x_norm, x_norm_orig, feat_mask):
    """
    Convert a normalised array (N, F) to raw feature values.

    For mutable integer features the rounding direction is decided relative to
    the original value so as not to bias the perturbation toward larger values:
      * perturbed ≥ original  →  ceil
      * perturbed < original →  floor
    """
    F = x_norm.shape[-1]
    x_raw = np.empty_like(x_norm)
    for fi in range(F):
        p = _SYM_PARAMS[fi]
        col = _inv_symlog(x_norm[:, fi], p)
        if feat_mask[fi] and FEAT_INTEGER[fi]:
            orig = _inv_symlog(x_norm_orig[:, fi], p)
            col = np.where(col >= orig, np.ceil(col), np.floor(col))
        x_raw[:, fi] = np.clip(col, p[0], p[-1])
    return x_raw


def from_raw_to_norm(x_raw, orig_shape, dtype, device):
    """Convert a raw array (N, F) back to a normalised tensor of *orig_shape*."""
    F = x_raw.shape[-1]
    x_norm = np.stack([_symlog(x_raw[:, fi], _SYM_PARAMS[fi]) for fi in range(F)], axis=1)
    return torch.tensor(x_norm.reshape(orig_shape), dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project(x_pert, x_ref, feat_mask, orig_shape, dtype, device):
    """
    Round-trip a perturbed normalised tensor through raw space and back.
    """
    F = orig_shape[-1]
    x_np = x_pert.cpu().numpy().reshape(-1, F)
    x_ref_np = x_ref.cpu().numpy().reshape(-1, F)
    x_raw = from_norm_to_raw(x_np, x_ref_np, feat_mask)
    return from_raw_to_norm(x_raw, orig_shape, dtype, device)