import yaml
import torch
from pathlib import Path

def load_config(config_path="../config.yaml"):
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: A dictionary containing configuration parameters.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f'Configuration file not found: {config_path}')
    
    with config_path.open('r') as file:
        config = yaml.safe_load(file)
    return config


def config_threads(n_thr=None):
    if n_thr:
        num_threads = torch.get_num_threads()
        print(f'WARNING: number of thread/s available: {num_threads}, using: {n_thr} thread/s')
        torch.set_num_threads(n_thr)
        torch.set_num_interop_threads(n_thr)