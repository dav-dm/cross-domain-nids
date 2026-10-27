import torch
from torch.utils.data import DataLoader


class InfiniteDataIterator:
    """
    A data iterator that will never stop producing data
    """
    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.device = device
        self._iterator = iter(self.data_loader)
    
    def __iter__(self):
        return self

    def __next__(self):
        data = self._get_next_batch()
        if self.device is not None:
            data = self.send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)

    def _get_next_batch(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.data_loader)
            return next(self._iterator)

    def send_to_device(self, data, device: torch.device):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, (list, tuple)):
            return type(data)(self.send_to_device(x, device) for x in data)
        elif isinstance(data, dict):
            return {k: self.send_to_device(v, device) for k, v in data.items()}
        else:
            return data
         