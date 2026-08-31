"""Snapshot integrity, public construction, and dataset-identity checks for CitrusT."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.torch_utils import get_flops


ROOT = Path(__file__).resolve().parents[1]
T_DIR = ROOT / "0_orange_yaml" / "T_series"
SOURCE_PAIRS = {
    "00_t00_yolo11n_reference.yaml": ROOT / "0_orange_yaml" / "A_baselines" / "current" / "001_yolo11-seg.yaml",
    "01_t01_f14_sppf_lska.yaml": ROOT / "0_orange_yaml" / "F_series" / "F14_yolo11-seg-sppf-lska.yaml",
    "02_t02_g10_hybrid.yaml": ROOT / "0_orange_yaml" / "G_series" / "10_yolo11n-seg-hybrid-lska-carafe-bifpn-p2b.yaml",
    "03_t03_n02_moce_lska.yaml": ROOT / "0_orange_yaml" / "N_series" / "02_moce_lska_carafe_p2b.yaml",
    "04_t04_l06_lska_topology.yaml": ROOT / "0_orange_yaml" / "L_series" / "06_lska_topology.yaml",
    "05_t05_s04_lite_head.yaml": ROOT / "0_orange_yaml" / "S_series" / "04_lite_head.yaml",
    "06_t06_b06_context_topology_lite.yaml": ROOT / "0_orange_yaml" / "B_series" / "06_b06_context_topology_lite.yaml",
    "07_t07_c03_dualproto_core.yaml": ROOT / "0_orange_yaml" / "C_series" / "03_c03_dualproto_core.yaml",
    "08_t08_d06_shape_semantic_full.yaml": ROOT / "0_orange_yaml" / "D_series" / "06_d06_shape_semantic_full.yaml",
    "09_t09_d07_deploy_lite.yaml": ROOT / "0_orange_yaml" / "D_series" / "07_d07_deploy_lite.yaml",
}


@pytest.mark.parametrize("yaml_path", sorted(T_DIR.glob("*.yaml")), ids=lambda path: path.stem)
def test_t_yaml_builds_through_public_api(yaml_path: Path) -> None:
    """Every selected model must build and run through the standard YOLO entry point."""
    wrapper = YOLO(str(yaml_path), task="segment", verbose=False)
    output = wrapper.model.eval()(torch.randn(1, 3, 64, 64))
    assert isinstance(output, tuple)
    assert wrapper.model.yaml["scale"] == "n"
    assert get_flops(wrapper.model, imgsz=64) > 0


@pytest.mark.parametrize("t_name,source", SOURCE_PAIRS.items(), ids=lambda item: Path(item).stem)
def test_t_snapshot_matches_source_graph(t_name: str, source: Path) -> None:
    """Comments and explicit scale locking may differ, but the executable graph must not drift."""
    snapshot = SegmentationModel(T_DIR / t_name, ch=3, nc=1, verbose=False)
    original = SegmentationModel(source, ch=3, nc=1, verbose=False)
    snapshot_modules = [module.__class__.__name__ for module in snapshot.model]
    original_modules = [module.__class__.__name__ for module in original.model]
    assert snapshot_modules == original_modules
    assert {key: value.shape for key, value in snapshot.state_dict().items()} == {
        key: value.shape for key, value in original.state_dict().items()
    }


def test_dataset_path_audit_when_local_fixture_available() -> None:
    """The batch runner must accept a user-provided data.yaml path."""
    data = ROOT.parents[2] / "data" / "orange_yolo_grouped_dedup_20260820" / "data.yaml"
    if not data.is_file():
        pytest.skip("Local clean dataset is not available")
    script = ROOT / "20260829_citrus_t_batch.py"
    spec = importlib.util.spec_from_file_location("citrus_t_batch", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    report, _ = module.audit_dataset(data)
    counts = tuple(report["splits"][split]["images"] for split in ("train", "val", "test"))
    assert counts == (676, 193, 96)
    assert not report["exact_path_overlap"]
