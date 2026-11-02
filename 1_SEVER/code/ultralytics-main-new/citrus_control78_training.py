"""Whole-image training with the historical 78 recipe and common EV6 validation."""
from citrus_e_v6_training import EV6TrainingTrainer
from ultralytics.models.yolo.segment import SegmentationTrainer


class Control78Trainer(EV6TrainingTrainer):
    """Use ordinary YOLO training data, NOT the inherited cached-view sampler."""

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode == "train":
            return SegmentationTrainer.build_dataset(self, img_path, mode, batch)
        return super().build_dataset(img_path, mode, batch)
