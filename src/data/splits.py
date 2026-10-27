import numpy as np
from sklearn.model_selection import train_test_split

from util.config import load_config


def train_val_test_split(dataset, seed):
    """
    Splits the dataset into training, validation, and test sets.
    """
    cf = load_config()
    
    arrays = [dataset['data'], dataset['labels'], dataset['quintuple']]

    # First split: entire dataset to train_val + test
    split1 = train_test_split(
        *arrays,
        train_size=cf['train_test_split'],
        random_state=seed,
        stratify=arrays[1]  # Always use full labels for stratification
    )
    train_val_arrays = split1[0::2]  # Every even-indexed element
    test_arrays = split1[1::2]       # Every odd-indexed element

    # Second split: train_val to train + val
    split2 = train_test_split(
        *train_val_arrays,
        train_size=cf['train_val_split'],
        random_state=seed,
        stratify=train_val_arrays[1]  # Stratify by train_val labels
    )
    train_arrays = split2[0::2]
    val_arrays = split2[1::2]

    return DataSplits(
        train=(train_arrays[0], train_arrays[1], train_arrays[2]), # data, labels, quintuple
        val=(val_arrays[0], val_arrays[1], val_arrays[2]),
        test=(test_arrays[0], test_arrays[1], test_arrays[2])
    )
    

class DataSplits:
    """
    A container for train/val/test splits.
    """
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test
        
        self.num_classes = len(np.unique(train[1]))