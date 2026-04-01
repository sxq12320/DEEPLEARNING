import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim         
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
from PIL import Image


class get_data(Dataset):
    def __init__(self , path):
        super().__init__()

        
        self.path = path


    def __len__(self):
        pass
    def __getitem__(self, index):
        return super().__getitem__(index)
    