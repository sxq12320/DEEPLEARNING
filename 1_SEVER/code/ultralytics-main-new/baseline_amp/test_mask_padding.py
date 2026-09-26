"""bug2(1) regression using official CPU padding code, even without compiled MMCV ops.

The AST loader isolates unmodified official classes; it does NOT emulate model
losses or CUDA operators. Full model steps run separately in server preflight.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mmdet_common import configure_mask_preprocessor


def official_cpu_classes():
    pytest.importorskip("mmengine")
    mmcv = pytest.importorskip("mmcv")
    spec = importlib.util.find_spec("mmdet")
    if spec is None:
        pytest.skip("Official mmdet wheel source unavailable")
    root = Path(spec.origin).parent
    import numpy as np
    import torch
    from abc import ABCMeta, abstractmethod
    from mmengine.model import ImgDataPreprocessor
    from mmengine.structures import BaseDataElement, InstanceData, PixelData
    from mmengine.utils import is_seq_of

    namespace = dict(np=np, torch=torch, nn=torch.nn, F=torch.nn.functional, mmcv=mmcv,
                     ABCMeta=ABCMeta, abstractmethod=abstractmethod, ImgDataPreprocessor=ImgDataPreprocessor,
                     BaseDataElement=BaseDataElement, InstanceData=InstanceData, PixelData=PixelData,
                     is_seq_of=is_seq_of)
    for relative, names in [
        ("structures/mask/structures.py", {"BaseInstanceMasks", "BitmapMasks"}),
        ("structures/det_data_sample.py", {"DetDataSample"}),
        ("models/data_preprocessors/data_preprocessor.py", {"DetDataPreprocessor"}),
    ]:
        path = root / relative
        selected = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(node, ast.ClassDef) and node.name in names]
        assert {node.name for node in selected} == names
        for node in selected:
            node.decorator_list = []  # Registry registration is not needed for isolated CPU checks.
        tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
                          + selected, type_ignores=[])
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    # Run the production synthetic-batch factory with the same official classes,
    # excluding only its imports that eagerly require unrelated compiled ops.
    path = HERE / "mmdet_smoke.py"
    factory = next(node for node in ast.parse(path.read_text(encoding="utf-8")).body
                   if isinstance(node, ast.FunctionDef) and node.name == "mixed_shape_batch")
    factory.body = [node for node in factory.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])), str(path), "exec"), namespace)
    return namespace


def test_original_failure_and_official_padding_fix(monkeypatch):
    import torch
    import mmdet_smoke

    classes = official_cpu_classes()
    factory, cls = classes["mixed_shape_batch"], classes["DetDataPreprocessor"]
    original = cls(boxtype2tensor=False)(factory(), training=True)
    tensors = [torch.from_numpy(s.gt_instances.masks.masks) for s in original["data_samples"]]
    with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
        torch.cat(tensors, dim=0)
    config = dict(data_preprocessor=dict(type="DetDataPreprocessor", mean=[103.53, 116.28, 123.675],
                                       std=[57.375, 57.12, 58.395], bgr_to_rgb=False))
    configure_mask_preprocessor(config)
    kwargs = dict(config["data_preprocessor"])
    kwargs.pop("type")
    # All synthetic boxes already are tensors. Other box-type conversion is not
    # part of this CPU-only test; server preflight uses the unmodified default.
    preprocessor = cls(**kwargs, boxtype2tensor=False)
    monkeypatch.setattr(mmdet_smoke, "mixed_shape_batch", factory)
    result = mmdet_smoke.check_mask_padding(preprocessor)
    assert result[0]["canvas"] == (640, 640)
    assert result[1]["canvas"] == (128, 128)


def test_mask_padding_preserves_normalization():
    config = dict(data_preprocessor=dict(type="DetDataPreprocessor", mean=[1, 2, 3],
                                       std=[4, 5, 6], bgr_to_rgb=True, pad_size_divisor=1))
    configure_mask_preprocessor(config)
    prep = config["data_preprocessor"]
    assert prep["mean"] == [1, 2, 3] and prep["std"] == [4, 5, 6]
    assert prep["bgr_to_rgb"] is True
    assert prep["pad_mask"] and prep["pad_size_divisor"] == 32 and prep["mask_pad_value"] == 0


def test_unknown_preprocessor_not_silently_overwritten():
    with pytest.raises(ValueError, match="Expected DetDataPreprocessor"):
        configure_mask_preprocessor(dict(data_preprocessor=dict(type="CustomPreprocessor")))
