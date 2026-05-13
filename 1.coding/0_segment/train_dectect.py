import torch

from configs.config import YOLO11_CONFIGS
from engine.losses import YOLODetectionLoss
from models import YOLO11Detector

cfg = YOLO11_CONFIGS["nano"]


model = YOLO11Detector(
    num_classes=80,
    reg_max=16,
    backbone_channels=cfg["channels"],
    depth_scale=cfg["depth_scale"],
)

x = torch.randn(2, 3, 640, 640)
cls_list, reg_list, features, neck_feats = model(x)
criterion = YOLODetectionLoss(num_classes=80, reg_max=16)
targets = [
    {
        "labels": torch.tensor([1, 2]),
        "boxes": torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.3, 0.3, 0.7, 0.7]]),
    },
    {"labels": torch.tensor([0]), "boxes": torch.tensor([[0.2, 0.2, 0.6, 0.6]])},
]
loss, items = criterion(cls_list, reg_list, targets, features)
