"""Export actual single-class J5 graph costs. No data access or training."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(output):
    import torch
    from citrus_j5_suite import KD, NAMES, YAML_DIR
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils.torch_utils import get_flops

    torch.set_num_threads(2)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in NAMES:
        model = SegmentationModel(YAML_DIR / f"{name}.yaml", nc=1, verbose=False).eval()
        row = dict(model=name, parameters=sum(p.numel() for p in model.parameters()),
                   gflops_640=get_flops(model, 640), mask_kd=KD[name]["kd_mask"], class_kd=KD[name]["kd_cls"])
        rows.append(row)
        print(json.dumps(row))
    with (output / "j5_single_view_costs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "cost_environment.json").write_text(json.dumps(dict(torch=torch.__version__,
        cuda_available=torch.cuda.is_available(), scope="THOP estimates; single-view @640; no runtime speed claim; "
            "teacher training cost excluded from deployment graph"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    main(**vars(parser.parse_args()))
