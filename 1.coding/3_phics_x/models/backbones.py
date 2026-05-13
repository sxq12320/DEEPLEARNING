import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    Basic_Conv_Block,
    Conv,
    Conv_Block_NONB,
)

class resnet_18(nn.Module):
