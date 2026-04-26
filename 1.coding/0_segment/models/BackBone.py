from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch.nn as nn
import torch 
import torch.nn.functional as F
import models
from configs.config import ACTIVATION_MAP
from utils.Block_function import get_activation , make_layers , autopad



# class ResNet_18(nn.Module):
#     def __init__(self):
#         super(ResNet_18 , self).__init__()
#         self.cfg = [
#             []
#         ]
#         self.ResNet_18_forward = make_layers(self.cfg)


if __name__ == "__main__":
    model = make_layers(
        cfg= [
            ['conv' , 3 , 64 , 3 , 1 , 1 , 1 , 1 ,False , 1]
        ]
    )
    print(model)