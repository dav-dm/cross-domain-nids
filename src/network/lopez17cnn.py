import math
import torch
import torch.nn.functional as F
from torch import nn

from module.head import FullyConnected
from network.base_network import BaseNetwork
from util.config import load_config


class Lopez17CNN(BaseNetwork):
    """
    From "Network Traffic Classifier With Convolutional and 
    Recurrent Neural Networks for Internet of Things"
    """
    def __init__(self, in_channels=1, **kwargs):
        super().__init__()

        num_classes = kwargs['num_classes']
        num_pkts = kwargs['num_pkts']
        num_fields = len(kwargs['fields'])
        self.out_features_size = 200

        kernel = (4, 2)
        stride = (1, 1)
        self.pool_kernel0 = (3, 2)
        self.pool_kernel1 = (3, 1) if num_fields < 6 else self.pool_kernel0
        self.pool_stride = (1, 1)

        for padding in ['valid', 'same']:
            features_size0, self.paddings0 = self._get_output_dim(
                num_pkts,
                kernels=[kernel[0], self.pool_kernel0[0], kernel[0], self.pool_kernel1[0]],
                strides=[stride[0], self.pool_stride[0], stride[0], self.pool_stride[0]],
                padding=padding,
                return_paddings=True
            )
            features_size1, self.paddings1 = self._get_output_dim(
                num_fields,
                kernels=[kernel[1], self.pool_kernel0[1], kernel[1], self.pool_kernel1[1]],
                strides=[stride[1], self.pool_stride[1], stride[1], self.pool_stride[1]],
                padding=padding,
                return_paddings=True
            )
            if features_size0 < 1 or features_size1 < 1:
                pass
            else:
                break

        cf = load_config()
        scaling_factor = cf['net_scale']
        filters = [math.ceil(32 * scaling_factor), math.ceil(64 * scaling_factor)]

        # Backbone
        self.backbone = nn.ModuleDict({
            'conv1': nn.Conv2d(in_channels, filters[0], kernel_size=kernel, stride=stride, padding=0),
            'bn1': nn.BatchNorm2d(filters[0]),

            'conv2': nn.Conv2d(filters[0], filters[1], kernel_size=kernel, stride=stride, padding=0),
            'bn2': nn.BatchNorm2d(filters[1]),

            'fc1': nn.Linear(features_size0 * features_size1 * filters[1], self.out_features_size),
        })
        # Init the network with a FullyConnected head
        self.set_head(FullyConnected(in_features=self.out_features_size, num_classes=num_classes))


    def forward(self, x, return_feat=False):
        embeddings = self.extract_features(x)
        out = F.relu(embeddings) # Activate the embeddings
        out = self.head(out)
        if return_feat:
            return out, embeddings
        return out

    def extract_features(self, x):
        out = F.pad(x, self.paddings1[0] + self.paddings0[0])
        out = F.relu(self.backbone['conv1'](out))
        out = F.pad(out, self.paddings1[1] + self.paddings0[1])
        out = F.max_pool2d(out, self.pool_kernel0, stride=self.pool_stride, padding=0)
        out = self.backbone['bn1'](out)
        
        out = F.pad(out, self.paddings1[2] + self.paddings0[2])
        out = F.relu(self.backbone['conv2'](out))
        out = F.pad(out, self.paddings1[3] + self.paddings0[3])
        out = F.max_pool2d(out, self.pool_kernel1, stride=self.pool_stride, padding=0)
        out = self.backbone['bn2'](out)
        
        out = torch.flatten(out, start_dim=1)
        out = self.backbone['fc1'](out)
        return out