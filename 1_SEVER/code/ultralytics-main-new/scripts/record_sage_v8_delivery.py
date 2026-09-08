"""Record measured initialization and delivery hashes without altering old experiments."""

# ruff: noqa: E402
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from citrus_sage_v8_suite import NAMES, YAML_DIR
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel


def main():
    torch.set_num_threads(2)
    output = ROOT / "reports/sage_v8_20260907/initialization"
    output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("v8_delivery", ROOT / "20260907_citrus_sage_v8_batch.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    checkpoint = ROOT / "yolo11n-seg.pt"
    callback = runner.initialization_recorder(checkpoint)
    source = YOLO(str(checkpoint)).model
    summary = []
    for name in NAMES:
        model = SegmentationModel(YAML_DIR / f"{name}.yaml", nc=1, verbose=False)
        model.load(source, verbose=False)
        directory = output / name
        directory.mkdir(exist_ok=True)
        callback(SimpleNamespace(model=model, save_dir=directory))
        record = json.loads((directory / "initialization_transfer.json").read_text())
        summary.append(dict(model=name, parameters=record["total_parameter_numel"],
                            initialized_equal_fraction=record["equal_fraction"]))
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    files = [
        ROOT / "ultralytics/nn/modules/citrus_sage_v8.py", ROOT / "ultralytics/nn/modules/__init__.py",
        ROOT / "ultralytics/nn/tasks.py", ROOT / "citrus_foreground.py", ROOT / "citrus_sage_v8_suite.py",
        ROOT / "20260907_citrus_sage_v8_batch.py", ROOT / "RUN_SAGE_V8.py", ROOT / "citrus_protocol.py",
        ROOT / "protocols/citrus_paper1_formal_v2_ram.yaml", ROOT / "ultralytics/utils/metrics.py",
        *YAML_DIR.glob("*.yaml"),
        Path("C:/Users/33836/Desktop/github/PiDiNet/models/ops.py"),
        Path("C:/Users/33836/Desktop/github/GSCNN/network/gscnn.py"),
        Path("C:/Users/33836/Desktop/Plug-play-modules-main/3. Block（功能模块）/(TPAMI 2024) FreqFusion.py"),
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (output.parent / "delivery_reference_sha256.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
