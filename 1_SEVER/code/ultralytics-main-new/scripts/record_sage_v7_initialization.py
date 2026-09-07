"""Record actual parameter transfer and selected source hashes for V7 delivery."""
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
from citrus_sage_v7_suite import NAMES, YAML_DIR
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel


def main():
    torch.set_num_threads(2)
    output = ROOT / "reports/sage_v7_20260906/initialization"
    output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("v7_init", ROOT / "20260906_citrus_sage_v7_batch.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    callback = runner.initialization_recorder(ROOT / "yolo11n-seg.pt")
    source = YOLO(str(ROOT / "yolo11n-seg.pt")).model
    summary = []
    for name in NAMES:
        model = SegmentationModel(YAML_DIR / f"{name}.yaml", nc=1, verbose=False)
        model.load(source, verbose=False)
        directory = output / name
        directory.mkdir(exist_ok=True)
        callback(SimpleNamespace(model=model, save_dir=directory))
        record = json.loads((directory / "initialization_transfer.json").read_text())
        summary.append(dict(model=name, params=record["total_parameter_numel"],
                            equal_fraction=record["equal_fraction"]))
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths = [ROOT / "ultralytics/nn/modules/citrus_sage_v7.py", ROOT / "ultralytics/nn/tasks.py",
             ROOT / "ultralytics/nn/modules/__init__.py", ROOT / "20260906_citrus_sage_v7_batch.py",
             ROOT / "RUN_SAGE_V7.py", ROOT / "citrus_sage_v7_suite.py", ROOT / "citrus_foreground.py",
             ROOT / "protocols/citrus_paper1_formal_v2_ram.yaml", *YAML_DIR.glob("*.yaml")]
    paths += [Path("C:/Users/33836/Desktop/github") / p for p in [
        "PKINet/mmrotate/models/backbones/pkinet.py", "TOOD/mmdet/models/dense_heads/tood_head.py",
        "QueryDet-PyTorch/models/querydet/det_head.py", "QueryDet-PyTorch/models/querydet/detector.py"]]
    paths += [Path("C:/Users/33836/Desktop/Plug-play-modules-main") / p for p in [
        "3. Block（功能模块）/(CVPR 2024) PKIBlock.py", "3. Block（功能模块）/(ECCV 2024) RCM.py",
        "3. Block（功能模块）/(TPAMI 2024) FreqFusion.py", "(arXiv 2023) RFAConv.py"]]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (output.parent / "delivery_and_reference_sha256.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
