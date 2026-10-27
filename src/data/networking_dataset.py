import torch
import hashlib
import numpy as np
from torch.utils.data import Dataset

from data.transformations import jittering


class NetworkingDataset(Dataset):
    def __init__(self, x, y, q, return_quintuple=False, apply_transform=False):
        super().__init__()
        self.x = x
        self.y = y
        self.q = q  # Quintuple of a biflow needed as random state to transform a biflow
        # Whether to return the quintuple along with the data and labels, used for explainability
        # Should be false in normal training and evaluation
        self.return_quintuple = return_quintuple 
        self.transform = None
        if apply_transform:
            self.transform = TwoViewsTrafficTransform(
                [jittering]
            )
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        q = self.q[idx]

        if self.transform is None and self.poisoning is None:
            result = (torch.from_numpy(x).float(), torch.from_numpy(np.array(y)).float())
            return result + (q,) if self.return_quintuple else result

        if self.transform is not None:
            x_weak, x_strong = self.transform(x, random_state=q)

            x_weak = torch.from_numpy(x_weak).float()
            x_strong = torch.from_numpy(x_strong).float()
            y = torch.from_numpy(np.array(y)).float()

            result = ((x_weak, x_strong), y)
            return result + (q,) if self.return_quintuple else result


class TwoViewsTrafficTransform:
    """
    TwoViewsTrafficTransform applies a random transformation from the provided list
    to generate two different views (weak and strong) of the same input biflow.
    """
    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, x, random_state):
        # Derive a seed integer from the random_state (quintuple)
        seed_int = int(hashlib.sha256(random_state.encode()).hexdigest(), 16) % (2**32)
        transformation = self.transformations[0]
        x_weak = transformation(x, a=0.5, random_state=seed_int)
        x_strong = transformation(x, a=2, random_state=seed_int + 1)
        return x_weak, x_strong