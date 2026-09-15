"""V6 common-raster validation; training raster remains an explicit ablation.

All validation polygons are rasterized freshly at input/2. Coarse prototype
logits are interpolated before thresholding, not binary masks after decoding.
Use EV6Validator for standalone comparisons with the same protocol.
"""

from copy import copy

from citrus_e_v5_slicing import MultiScaleTrainingTrainer
from ultralytics.data import build_yolo_dataset
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils.torch_utils import unwrap_model


class EV6Validator(SegmentationValidator):
    """Evaluate every V6 arm against the same input/2 GT raster."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = copy(self.args)
        self.args.mask_ratio = 2
        self.evaluation_mask_ratio = 2


class EV6TrainingTrainer(MultiScaleTrainingTrainer):
    """Keep V5 multi-scale input; decouple training and validation mask grids."""

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode == "train":
            return super().build_dataset(img_path, mode, batch)
        cfg = copy(self.args)
        cfg.mask_ratio = 2
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_yolo_dataset(cfg, img_path, batch, self.data, mode=mode, rect=True, stride=stride)

    def get_validator(self):
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"
        return EV6Validator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
