import numpy as np

from data.data_module import DataModule
from data.splits import DataSplits
from data.reader import get_data_labels
from data.splits import train_val_test_split


class DataManager:
    """
    Manages and prepares data for training, validation, and testing.
    """
    def __init__(self, args):
        self.args = args
        self.seed = self.args.seed
        
    def prepare_datasets(self):
        dataset_args = {
            'num_pkts': self.args.num_pkts,
            'fields': self.args.fields,
            'is_flat': self.args.is_flat,
            'seed': self.args.seed,
        }
                    
        src_dataset = get_data_labels(dataset=self.args.src_dataset, **dataset_args)
        self.src_splits = train_val_test_split(dataset=src_dataset, seed=self.seed)
        self.num_classes = self.src_splits.num_classes
        
        if self.args.trg_dataset is not None:
            trg_dataset = get_data_labels(dataset=self.args.trg_dataset, **dataset_args)
            self.trg_splits = train_val_test_split(dataset=trg_dataset, seed=self.seed)
            
            if self.src_splits.num_classes != self.trg_splits.num_classes:
                raise ValueError(
                    'Mismatch between the classes of the source and target datasets'
                )
        else:
            self.trg_splits = None
            
            
    def get_datamodule(self, split='src'):
        """
        Returns a DataModule for the specified dataset split ('src' or 'trg').
        """
        if split == 'src':
            splits = self.src_splits
        elif split == 'trg':
            splits = self.trg_splits
        else:
            raise ValueError("Split must be either 'src' or 'trg'")
        
        return DataModule(
            train_dataset=splits.train, # Includes data, labels, quintuple
            val_dataset=splits.val,
            test_dataset=splits.test,
            **vars(self.args)
        )
        
        
    def concat_src_trg(self, unsup_trg=False):
        """
        Concatenates source and target datasets for training and validation.
        If unsup_trg is True, target training data is treated as unlabeled.
        """        
        combined_train = self._concat_dataset(
            self.src_splits.train,
            self.trg_splits.train,
            unsup='d2' if unsup_trg else None
        )
        combined_val = self._concat_dataset(
            self.src_splits.val,
            self.trg_splits.val
        )
        combined_splits = DataSplits(
            train=combined_train,
            val=combined_val,
            test=None
        )
        return DataModule(
            train_dataset=combined_splits.train,
            val_dataset=combined_splits.val,
            test_dataset=None,
            **vars(self.args)
        )     
        
        
    def get_anomaly_datamodule(self):
        x_train, y_train, q_train = self.src_splits.train
        if len(np.unique(y_train)) > 2:
            raise ValueError(
                "More than 2 classes found in the source dataset. "
                "This function is intended for binary classification with benign samples labeled as 0.")
            
        ben_mask = (y_train == 0)
        train_ben = (x_train[ben_mask], y_train[ben_mask], np.array(q_train)[ben_mask])

        combined_val = self._concat_dataset(
            self.src_splits.val,
            self.trg_splits.val
        )

        return DataModule(
            train_dataset=train_ben,
            val_dataset=combined_val,
            test_dataset=None,
            **vars(self.args)
        )
    
    
    def _concat_dataset(self, d1, d2, unsup=None):
        x1, y1, q1 = d1
        x2, y2, q2 = d2
        
        if unsup == 'd1':
            y1 = np.full_like(y1, -1)
        elif unsup == 'd2':
            y2 = np.full_like(y2, -1)
        
        return np.concatenate([x1, x2]), np.concatenate([y1, y2]), np.concatenate([q1, q2]) 