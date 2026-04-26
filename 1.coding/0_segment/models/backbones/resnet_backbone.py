from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch.nn as nn
import torch 
import torch.nn.functional as F
from configs.config import ACTIVATION_MAP,RESNET_18_CFG
from utils.common import get_activation , autopad
from ..builders.builder import make_layers


# class ResNet_18(nn.Module):
#     def __init__(self):
#         super(ResNet_18 , self).__init__()
#         self.cfg = [
#             []
#         ]
#         self.ResNet_18_forward = make_layers(self.cfg)


if __name__ == "__main__":
    model = make_layers(
        cfg=RESNET_18_CFG
    )
    print(model)