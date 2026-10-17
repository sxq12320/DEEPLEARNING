"""D3: real EV6 augmented batches, frozen checkpoint and TAL, no parameter update.

Original data remains untouched. Prepared crops and label caches are private
diagnostic derivatives. Hook exposure counts are NOT unique source-instance
attrition rates. Final masks, deleted transformation exposures, and assignment
have separate denominators.
"""

# ruff: noqa: E402
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
import yaml

from citrus_e_v5_slicing import prepare_multiscale_views
from citrus_e_v6_training import EV6TrainingTrainer
from citrus_slicing import source_files
from ultralytics import YOLO
from ultralytics.data import build_dataloader
from ultralytics.data.augment import RandomPerspective
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.instance import Instances
from ultralytics.utils.torch_utils import init_seeds


def state_digest(model):
    """Hash parameters and buffers, checking that even BN statistics are frozen."""
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        result.update(name.encode())
        result.update(tensor.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new diagnostic output")
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    init_seeds(42, deterministic=True)
    prepared = prepare_multiscale_views(args.data, args.prepared, 640)
    data, _ = source_files(prepared, "train")
    recorded_args = yaml.safe_load((args.weights.parent.parent / "args.yaml").read_text(encoding="utf-8"))
    # Retain the weight's actual historical augmentation settings; local worker
    # count is zero for instrumentation, not a modification to training recipes.
    hyp = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, **recorded_args, "workers": 0, "cache": True})
    trainer = object.__new__(EV6TrainingTrainer)
    trainer.args, trainer.data = hyp, data
    dataset = trainer.build_dataset(data["train"], mode="train", batch=args.batch)
    loader = build_dataloader(dataset, batch=args.batch, workers=0, shuffle=True, rank=-1)
    model = YOLO(str(args.weights)).model.float().eval()
    model.args = hyp
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    digest_before = state_digest(model)
    criterion = model.init_criterion()
    exposure_rows, mask_rows, assignment_rows = [], [], []
    original_candidates = RandomPerspective.box_candidates
    original_zero = Instances.remove_zero_area_boxes
    current_batch = [0]

    def track_candidates(box1, box2, **kwargs):
        keep = original_candidates(box1, box2, **kwargs)
        for k, good in enumerate(keep):
            exposure_rows.append(dict(batch=current_batch[0], stage="perspective",
                                      transformed_short=float(min(box2[2, k] - box2[0, k], box2[3, k] - box2[1, k])),
                                      before_scaled_short=float(min(box1[2, k] - box1[0, k], box1[3, k] - box1[1, k])),
                                      kept=bool(good)))
        return keep

    def track_zero(instances):
        count = len(instances)
        good = original_zero(instances)
        for k in range(count):
            exposure_rows.append(dict(batch=current_batch[0], stage="zero_area_clip", kept=bool(good[k])))
        return good

    def track_assignment(_module, inputs, output):
        valid = inputs[5][..., 0].bool()
        fg, ids = output[3], output[4]
        for image in range(valid.shape[0]):
            counts = torch.bincount(ids[image, fg[image]].long(), minlength=valid.shape[1])
            for gt in torch.where(valid[image])[0].tolist():
                assignment_rows.append(dict(batch=current_batch[0], image=image, gt=gt,
                                            positive_anchors=int(counts[gt])))

    RandomPerspective.box_candidates = staticmethod(track_candidates)
    Instances.remove_zero_area_boxes = track_zero
    hook = criterion.assigner.register_forward_hook(track_assignment)
    started = time.perf_counter()
    iterator = iter(loader)
    try:
        for batch_number in range(args.batches):
            current_batch[0] = batch_number
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            h, w = batch["img"].shape[-2:]
            for image, file in enumerate(batch["im_file"]):
                boxes = batch["bboxes"][batch["batch_idx"] == image]
                raster = batch["masks"][image]
                for gt, box in enumerate(boxes):
                    mask_rows.append(dict(batch=batch_number, image=image, gt=gt,
                                          prepared_view=Path(file).name,
                                          short=float(min(box[2] * w, box[3] * h)),
                                          visible_area_input=int((raster == gt + 1).sum()) * 4,
                                          empty_mask=not bool((raster == gt + 1).any())))
            batch["img"] = batch["img"].float() / 255
            with torch.inference_mode():
                predictions = model(batch["img"])[1]
                criterion.get_assigned_targets_and_loss(predictions, batch)
            print(f"D3 {batch_number + 1}/{args.batches}, elapsed={time.perf_counter() - started:.1f}s", flush=True)
    finally:
        RandomPerspective.box_candidates = staticmethod(original_candidates)
        Instances.remove_zero_area_boxes = original_zero
        hook.remove()
    assigned_lookup = {(r["batch"], r["image"], r["gt"]): r["positive_anchors"] for r in assignment_rows}
    for row in mask_rows:
        row["positive_anchors"] = assigned_lookup.get((row["batch"], row["image"], row["gt"]), 0)
    digest_after = state_digest(model)
    assert digest_before == digest_after, "Frozen probe must not change parameters or buffers"
    for name, records in (("transform_exposures", exposure_rows), ("retained_gt", mask_rows)):
        with (args.output / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in records))))
            writer.writeheader()
            writer.writerows(records)

    def summarize(rows):
        n = len(rows)
        return dict(gt=n, zero_positive=sum(r["positive_anchors"] == 0 for r in rows),
                    zero_mask=sum(r["empty_mask"] for r in rows),
                    short_lt4=sum(r["short"] < 4 for r in rows), short_lt2=sum(r["short"] < 2 for r in rows),
                    mean_positives=sum(r["positive_anchors"] for r in rows) / n if n else None)

    summary = dict(evidence="旧划分探索", weights=str(args.weights), data=str(args.data),
                   batches=args.batches, batch=args.batch, actual_cache=dataset.cache,
                   state_before=digest_before, state_after=digest_after,
                   retained=summarize(mask_rows),
                   scale_summary=dataset.scale_summary,
                   bins={f"{lo}-{hi}": summarize([r for r in mask_rows if lo <= r["short"] < hi])
                         for lo, hi in ((0, 8), (8, 16), (16, 32), (32, 64), (64, 100000))},
                   exposures={stage: dict(total=sum(r["stage"] == stage for r in exposure_rows),
                                          removed=sum(r["stage"] == stage and not r["kept"] for r in exposure_rows))
                              for stage in ("perspective", "zero_area_clip")},
                   augmentation={k: getattr(hyp, k) for k in ("mosaic", "scale", "copy_paste", "mask_ratio")},
                   limits=["Frozen end-of-training I07 predictions, not early training dynamics or scratch J20.",
                           "Deleted transformation exposures include clipping/outside pixels, not necessarily harmful removal.",
                           "No complete source-GT lineage through donor/mosaic/copy-paste; no unique-source loss percentage.",
                           "A retained <2px denominator cannot expose filtered GT; separate exposure CSV is required."])
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
