import numpy as np

def jittering(x, a=1.0, u=0, s=.03, features='all', random_state=None):
    """
    Add random noise to the input biflow.
    The parameter `a` scales the strength of the jittering.
    """
    if isinstance(features, str) and features == 'all':
        features = list(range(x.shape[-1]))

    mask = np.zeros_like(x)
    padding_mask = np.all(x == np.array([0,0,0.5,0,0,0]), axis=-1)
    
    for f in features:
        valid_idx = ~padding_mask
        feature_values = x[..., f]
        scale_noise = s * a * np.abs(feature_values) # Noise is weighted w.r.t. feature value magnitude
        noise = _execute_revert_random_state(
            np.random.normal,
            dict(loc=u, scale=scale_noise),
            new_random_state=random_state
        )
        mask[..., f][valid_idx] = noise[valid_idx]
    
    mask[:, 0, 1] = 0
    x_tr = np.abs(x + mask).astype(x.dtype)
    x_tr[x_tr > 1] = 1
    
    return x_tr


def _execute_revert_random_state(fn, fn_kwargs=None, new_random_state=None):
    """
    Execute fn(**fn_kwargs) without impacting the external random_state behavior.
    """
    old_random_state = np.random.get_state()
    np.random.seed(new_random_state)
    ret = fn(**fn_kwargs)
    np.random.set_state(old_random_state)
    return ret