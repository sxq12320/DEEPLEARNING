"""Same-checkpoint EV6 ablations: distinguish quality rescoring from pair selection.

Runs in the foreground. Nothing retrains, changes labels, changes EV6 matching,
or marks an untrained relation head as trained. Use a new project per evaluation.
"""

import argparse
import csv
import json
from pathlib import Path


def apply_mode(head, mode):
    """Apply an explicit inference-only intervention to a fresh checkpoint copy."""
    if mode == "quality_only":
        if not head.enabled or not head.selection:
            raise ValueError("quality_only requires a checkpoint trained with selection=True")
        head.relation.mark_trained(False)
    elif mode == "global":
        head.enabled = False
    elif mode == "no_gain":
        head.gain_route = False
    elif mode == "no_neighbor":
        head.neighbor = False
    elif mode != "full":
        raise ValueError(f"Unknown J6 inference intervention: {mode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="1")
    parser.add_argument("--project", required=True)
    parser.add_argument("--modes", default="full,quality_only,global")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    allowed = {"full", "quality_only", "global", "no_gain", "no_neighbor"}
    if not modes or len(set(modes)) != len(modes) or set(modes) - allowed:
        parser.error("Choose distinct modes from full,quality_only,global,no_gain,no_neighbor")
    if args.batch < 1 or args.workers < 0:
        parser.error("Positive batch and nonnegative workers required")
    from RUN_CITRUS_J6 import configure_device

    configure_device(args.device, args.cpu_threads)
    import torch
    from citrus_j6_training import find_j6_head
    from ultralytics import YOLO

    torch.set_num_threads(args.cpu_threads)
    weights, data, project = (Path(value).expanduser().resolve() for value in
                              (args.weights, args.data, args.project))
    if not weights.is_file() or not data.is_file():
        raise FileNotFoundError("Both --weights and --data must exist")
    if project.exists():
        raise FileExistsError(f"Choose a NEW evaluation project; refusing to overwrite {project}")
    if not args.no_plots:
        from citrus_j6_batch import plotting_preflight

        plotting_preflight()
    project.mkdir(parents=True)
    rows = []
    for mode in modes:
        model = YOLO(str(weights), task="segment")
        head = find_j6_head(model.model)
        original_readiness = bool(getattr(head, "relation", None) is not None and head.relation.relation_trained)
        apply_mode(head, mode)
        print(f"J6 EV6 mode={mode}; saved relation trained={original_readiness}; "
              "same checkpoint, no retraining, input/2 metric raster", flush=True)
        if mode == "full" and head.selection and not original_readiness:
            print("WARNING: saved pair relation is not supervised-ready; full will NOT force random suppression.", flush=True)
        metrics = model.val(data=str(data), device=args.device, project=str(project), name=mode,
                            split=args.split, imgsz=640, batch=args.batch, workers=args.workers,
                            mask_ratio=2, conf=.001, iou=.7, max_det=300, half=False,
                            plots=not args.no_plots, save_json=False, augment=False, compile=False)
        row = dict(mode=mode, saved_relation_ready=original_readiness,
                   mask_AP5095=float(metrics.seg.map), mask_AP50=float(metrics.seg.map50),
                   mask_precision=float(metrics.seg.mp), mask_recall=float(metrics.seg.mr))
        rows.append(row)
        record = dict(scope="Inference intervention, not a new independently trained architecture",
                      weights=str(weights), data=str(data), split=args.split, mode=mode,
                      input=640, evaluation="unchanged I/E/J EV6 input/2", conf=.001, iou=.7,
                      max_det=300, half=False, plots=not args.no_plots, metrics=row)
        (project / mode / "j6_intervention.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    with (project / "j6_interventions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("J6 evaluation stopped; no next mode starts.", flush=True)
        raise SystemExit(130)
