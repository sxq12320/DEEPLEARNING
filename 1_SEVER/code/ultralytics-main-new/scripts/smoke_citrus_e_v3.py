"""Explicit TEST ONLY four-image copy. Run real E V3 batch for one epoch, no protocol override."""

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    import yaml

    from citrus_slicing import source_files
    from ultralytics.data.utils import img2label_paths

    torch.set_num_threads(2)
    fixture = args.output / "SMOKE_NOT_FORMAL_DATA"
    for split, number in (("train",4),("val",2)):
        config, files = source_files(args.data,split)
        for directory in ("images","labels"):
            (fixture/split/directory).mkdir(parents=True)
        for file in files[:number]:
            source = Path(file)
            shutil.copy2(source,fixture/split/"images"/source.name)
            label = Path(img2label_paths([str(source)])[0])
            shutil.copy2(label,fixture/split/"labels"/label.name)
    data = fixture / "data.yaml"
    data.write_text(yaml.safe_dump(dict(path=str(fixture.resolve()),train="train/images",val="val/images",
                                       names=config["names"])),encoding="utf-8")
    spec = importlib.util.spec_from_file_location("ev3_smoke_runner",ROOT/"20260909_citrus_e_v3_batch.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    sys.argv = [str(spec.origin),"--data",str(data),"--suite","smoke","--epochs","1","--device","cpu",
                "--project",str(args.output/"SMOKE_NOT_FORMAL_RUNS"),"--fail-fast"]
    runner.main()


if __name__ == "__main__":
    main()
