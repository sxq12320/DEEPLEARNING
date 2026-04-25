import torch.nn as nn
import torch 
import torch.nn.functional as F
import models
from configs.config import ACTIVATION_MAP
from utils.Block_function import get_activation , make_layers , autopad



class ResNet_18(nn.Module):
    def __init__(self):
        super(ResNet_18 , self).__init__()
        self.cfg = [
            []
        ]
        self.ResNet_18_forward = make_layers(self.cfg)

        