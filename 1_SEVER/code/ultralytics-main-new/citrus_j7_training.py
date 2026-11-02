"""Native-image J7 training with the I/J EV6 raster and mask-selected final evaluation."""

import hashlib
import json
from copy import copy
from pathlib import Path

from citrus_e_v6_training import EV6Validator
from citrus_j_batch import save_mask_checkpoints
from citrus_j7_pr_audit import J7FinalAuditValidator
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils import LOGGER


class J7Trainer(SegmentationTrainer):
    """Keep native YOLO images/losses; select final plots using mask AP, not box-plus-mask fitness.

    The normal ``best.pt`` selection is preserved for auditing. This trainer neither creates
    sliced/paired training views nor uses the cross-framework baseline evaluator.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.args.batch, int) or self.args.batch < 1:
            raise ValueError("J7 controlled experiments require a fixed positive integer batch, not auto-batch")
        self._j7_requested_batch = self.args.batch
        self.add_callback("on_model_save", save_mask_checkpoints)

    def _build_train_pipeline(self):
        """Do not let upstream OOM recovery silently change the comparison's batch size."""
        if self.args.batch != self._j7_requested_batch:
            raise RuntimeError(
                "The trainer attempted to reduce batch after an OOM. J7 stopped to preserve the fixed protocol; "
                "choose a smaller common BATCH and a NEW PROJECT for every compared arm."
            )
        return super()._build_train_pipeline()

    def get_validator(self):
        """Use the same input/2 ground-truth raster as the I/J experiments."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"
        return EV6Validator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def final_eval(self):
        """Re-evaluate ``best_mask.pt`` separately, retaining resumable training checkpoints."""
        selected = Path(self.save_dir) / "weights" / "best_mask.pt"
        selection_file = Path(self.save_dir) / "best_mask_selection.json"
        if not selected.is_file() or not selection_file.is_file():
            raise RuntimeError(
                "Mask-selected checkpoint is missing. J7 requires val=True and save=True; "
                "best.pt is deliberately NOT substituted for best_mask.pt."
            )
        selection = json.loads(selection_file.read_text(encoding="utf-8"))
        output = Path(self.save_dir) / "selected_mask_validation"
        if output.exists():
            raise FileExistsError("Refusing to overwrite selected validation output: {}".format(output))
        args = copy(self.args)
        args.half = False
        args.compile = False
        args.plots = bool(self.args.plots)
        args.save_json = False
        args.save_txt = False
        args.split = "val"
        LOGGER.info("\nJ7 final validation: best_mask.pt, epoch {}, EV6 raster".format(selection["epoch"]))
        validator = J7FinalAuditValidator(
            self.test_loader, save_dir=output, args=args, _callbacks=self.callbacks
        )
        self.metrics = validator(model=str(selected))
        self.metrics.pop("fitness", None)
        self.validator = validator
        record = {
            "checkpoint": str(selected.resolve()),
            "native_fitness_checkpoint": str(Path(self.best).resolve()),
            "checkpoint_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
            "selection": selection,
            "validator": "EV6Validator",
            "mask_ratio": 2,
            "half": False,
            "metrics": self.metrics,
            "plots": bool(args.plots),
            "note": "Native best.pt is retained but is not the source of these metrics/plots.",
        }
        (output / "selected_checkpoint.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        self.j7_final_eval_record = record
        # Model.train() reloads trainer.best after this method. Redirect that in-memory
        # pointer too, so its returned model and metrics refer to the SAME checkpoint.
        # The native weights/best.pt file itself remains intact for auditing.
        self.j7_native_best = self.best
        self.best = selected
        # Unlike the upstream finalizer, do not strip last.pt in place: its optimizer
        # state is useful if a run is interrupted just before this final evaluation.
        self.epoch += 1
        try:
            self.run_callbacks("on_fit_epoch_end")
        finally:
            self.epoch -= 1
