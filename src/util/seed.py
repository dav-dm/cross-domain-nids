import os
import random
import numpy as np
import torch

from util.config import load_config


def seed_everything(seed):
    """
    From Pytorch Lightining
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    In addition, sets the env variable `PL_GLOBAL_SEED` which will be passed to
    spawned subprocesses (e.g. ddp_spawn backend).

    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
    """
    cf = load_config()
    
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    try:
        if seed is None:
            seed = os.environ.get("PL_GLOBAL_SEED", _select_seed_randomly(min_seed_value, max_seed_value))
        seed = int(seed)
    except (TypeError, ValueError):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    if (seed > max_seed_value) or (seed < min_seed_value):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = cf['deterministic']
    torch.backends.cudnn.benchmark = cf['benchmark']
    return seed

def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    seed = random.randint(min_seed_value, max_seed_value)
    return seed