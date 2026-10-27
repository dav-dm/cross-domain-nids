import torch.nn as nn


class FullyConnected(nn.Module):
    """
    Fully-connected (linear) head
    """
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.fc(x)
