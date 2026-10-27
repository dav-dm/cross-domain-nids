import numpy as np
import torch
from torch import nn
from argparse import ArgumentParser
from torch.autograd import Function

from util.config import load_config


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx, input, coeff):
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    """
    A PyTorch module that implements a warm-start gradient reversal layer. This layer reverses 
    the gradient during backpropagation with a coefficient that changes dynamically based on 
    the training iteration. It is commonly used in domain adaptation tasks.
    [[Link to Source Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/modules/grl.py)
    
    Args:
        alpha (float): Scaling factor for the gradient reversal coefficient. Default is 1.0.
        lo (float): Minimum value of the coefficient. Default is 0.0.
        hi (float): Maximum value of the coefficient. Default is 1.0.
        max_iters (int): Maximum number of iterations for coefficient adjustment. Default is 1000.
        auto_step (bool): Whether to automatically increment the iteration counter. Default is False.
    """
    def __init__(self, **kwargs):
        super(WarmStartGradientReverseLayer, self).__init__()
        cf = load_config()
        
        self.alpha = kwargs.get('wsgrl_alpha', cf['wsgrl_alpha'])
        self.lo = kwargs.get('wsgrl_lo', cf['wsgrl_lo'])
        self.hi = kwargs.get('wsgrl_hi', cf['wsgrl_hi'])
        self.max_iters = kwargs.get('wsgrl_max_iters', cf['wsgrl_max_iters'])
        self.auto_step = kwargs.get('wsgrl_auto_step', cf['wsgrl_auto_step'])
        self.iter_num = 0
        
    @staticmethod
    def add_specific_args(parent_parser):
        cf = load_config()
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=True,
            conflict_handler='resolve',
        )
        parser.add_argument('--wsgrl-alpha', type=float, default=cf['wsgrl_alpha'])
        parser.add_argument('--wsgrl-lo', type=float, default=cf['wsgrl_lo'])
        parser.add_argument('--wsgrl-hi', type=float, default=cf['wsgrl_hi'])
        parser.add_argument('--wsgrl-max-iters', type=int, default=cf['wsgrl_max_iters'])
        parser.add_argument('--wsgrl-auto-step', action='store_true', default=cf['wsgrl_auto_step'])
        return parser
        

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        self.iter_num += 1
