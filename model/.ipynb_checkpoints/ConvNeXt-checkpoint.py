import torch
from torch import nn
from torchvision.ops.misc import Permute
from torchvision.ops import StochasticDepth

class ConvBlock(nn.Module):
    def __init__(self, in_channels, init_layer_scale, p_stochastic_depth):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels), # Depthwise Convolution
            Permute([0, 2, 3, 1]),
            nn.LayerNorm(in_channels, eps=1e-6),
            Permute([0, 3, 1, 2]),
            nn.Conv2d(in_channels, 4*in_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(4*in_channels, in_channels, kernel_size=1)
        )

        self.layer_scale = nn.Parameter(torch.ones(1, in_channels, 1, 1) * init_layer_scale)
        self.stochastic_depth = StochasticDepth(p_stochastic_depth, "row")

    def forward(self, x):
        residual = self.layer_scale * self.residual(x)
        residual = self.stochastic_depth(residual)
        x = x + residual
        
        return x


class ConvNeXt(nn.Module):
    def __init__(self, block_setting, init_layer_scale=1e-6, p_stochastic_depth=0.0, num_classes=1000, **kwargs):
        super().__init__()


    def forward(self):
        