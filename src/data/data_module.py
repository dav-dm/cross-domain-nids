import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from data.networking_dataset import NetworkingDataset
from util.config import load_config


class DataModule:
    def __init__(self, train_dataset, val_dataset, test_dataset=None, **kwargs):
        cf = load_config()
        self.batch_size = kwargs.get('batch_size', cf['batch_size'])
        self.adapt_batch_size = kwargs.get('adapt_batch_size', cf['adapt_batch_size'])
        self.num_workers = kwargs.get('num_workers', cf['num_workers'])
        self.pin_memory = kwargs.get('pin_memory', cf['pin_memory'])
        self.drop_last = kwargs.get('drop_last', cf['drop_last'])
        self.apply_transformations = kwargs.get('apply_transformations', False)

        self.train_x, self.train_y, self.train_q = train_dataset
        self.val_x, self.val_y, self.val_q = val_dataset
        self.test_x, self.test_y, self.test_q = test_dataset or (None, None, None)
        
        self.approach_type = kwargs.get('appr_type', None)
        self.seed = kwargs.get('seed', cf['seed'])
        
        self.return_quintuple = kwargs.get('return_quintuple', cf['return_quintuple'])
        
        
    @staticmethod
    def add_argparse_args(parent_parser):
        cf = load_config()
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=True,
            conflict_handler='resolve',
        )
        parser.add_argument('--batch-size', type=int, default=cf['batch_size'])
        parser.add_argument('--adapt-batch-size', type=int, default=cf['adapt_batch_size'])
        parser.add_argument('--num-workers', type=int, default=cf['num_workers'])
        parser.add_argument('--pin-memory', action='store_true', default=cf['pin_memory'])
        parser.add_argument('--drop-last', action='store_true', default=cf['drop_last'])
        parser.add_argument('--return-quintuple', action='store_true', default=cf['return_quintuple'])
        return parser
    
    def set_train_dataset(self, dataset):
        self.train_x, self.train_y, self.train_q = dataset
        
    def set_val_dataset(self, dataset):
        self.val_x, self.val_y, self.val_q = dataset
        
    def set_test_dataset(self, dataset):
        self.test_x, self.test_y, self.test_q = dataset

    def get_train_data(self):
        
        if self.approach_type == 'dl':     
            return DataLoader(
                NetworkingDataset(self.train_x, self.train_y, self.train_q),
                batch_size=self.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(self.seed),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
            )
        else:
            # For ML approaches
            return self.train_x, self.train_y
        
        
    def get_val_data(self):
        
        if self.approach_type == 'dl':            
            return DataLoader(
                NetworkingDataset(self.val_x, self.val_y, self.val_q),
                batch_size=self.batch_size,
                shuffle=False,
                generator=torch.Generator().manual_seed(self.seed),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        else:
            # For ML approaches
            return self.val_x, self.val_y   
        
        
    def get_test_data(self):
        
        if self.approach_type == 'dl':
            return DataLoader(
                NetworkingDataset(self.test_x, self.test_y, self.test_q,
                                  return_quintuple=self.return_quintuple),
                batch_size=self.batch_size,
                shuffle=False,
                generator=torch.Generator().manual_seed(self.seed),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        else:
            # For ML approaches
            return self.test_x, self.test_y
        
        
    def get_adapt_data(self):
        
        if self.approach_type == 'dl':
            return DataLoader(
                NetworkingDataset(self.train_x, self.train_y, self.train_q, 
                                  apply_transform=self.apply_transformations),
                batch_size=self.adapt_batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(self.seed),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
            )
        else:
            # For ML approaches
            return self.train_x, self.train_y
        