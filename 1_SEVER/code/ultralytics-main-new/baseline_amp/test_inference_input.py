"""CPU contracts for bounded input preparation and unchanged one-image inference."""

# ruff: noqa: E402 -- fixtures intentionally select the isolated helper import path

import ast
import collections.abc
import json
import sys
import threading
import types
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pytest
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import inference_input
import validation_adapters as adapters
from inference_input import OrderedInputPredictor
from registry import COMMON_VALIDATION


def same_predictions(actual, expected):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        for got, wanted in zip(left, right):
            assert np.array_equal(np.asarray(got), np.asarray(wanted))


def make_images(tmp_path):
    paths = []
    for index, shape in enumerate(((23, 31), (39, 17), (18, 42))):
        rows, columns = np.indices(shape)
        image = np.stack(((rows * 7 + index) % 256, (columns * 11) % 256, (rows + columns * 3) % 256), axis=-1)
        path = tmp_path / f"{index}.png"
        Image.fromarray(image.astype(np.uint8)).save(path)
        paths.append(path)
    return paths


def test_prefetch_preserves_order_and_runs_inference_in_consumer_thread():
    consumer = threading.get_ident()
    second_ready = threading.Event()
    preparations, completions, inferences = [], [], []

    def prepare(value):
        preparations.append(threading.get_ident())
        if value == 0:
            assert second_ready.wait(5)
        elif value == 1:
            completions.append(value)
            second_ready.set()
            return value * 10
        completions.append(value)
        return value * 10

    def infer(value):
        inferences.append(threading.get_ident())
        return value + 1

    predictor = OrderedInputPredictor(prepare, infer, prefetch_workers=2)
    assert list(predictor.iter_predictions(range(8))) == [value * 10 + 1 for value in range(8)]
    assert completions.index(1) < completions.index(0)
    assert all(identity != consumer for identity in preparations)
    assert inferences == [consumer] * 8
    assert set(predictor.last_timing) == {"prepare_seconds", "input_wait_seconds", "inference_seconds"}
    assert all(value >= 0 for value in predictor.last_timing.values())


@pytest.mark.parametrize("workers", [0, 1, 2, 12])
@pytest.mark.parametrize("bound", [2, 4, 12])
def test_prefetch_is_bounded_and_never_consumes_entire_source(workers, bound):
    consumed = []
    inferred = []

    def paths():
        for value in range(100):
            consumed.append(value)
            yield value

    def infer(value):
        inferred.append(value)
        assert len(consumed) <= len(inferred) - 1 + (min(4, bound) if workers else 1)
        return value

    predictor = OrderedInputPredictor(lambda value: value, infer, workers, bound)
    iterator = predictor.iter_predictions(paths())
    assert next(iterator) == 0
    assert len(consumed) == (min(4, bound) if workers else 1)
    assert next(iterator) == 1
    iterator.close()
    assert len(consumed) < 100
    assert predictor.prefetch_workers <= predictor.max_prefetch <= 4


@pytest.mark.parametrize("workers", [0, 2])
def test_empty_input_does_not_prepare_or_infer(workers):
    def unexpected(value):
        pytest.fail("Empty input must not prepare or infer")

    assert list(OrderedInputPredictor(unexpected, unexpected, workers).iter_predictions([])) == []


def test_zero_workers_and_call_are_synchronous_without_a_pool(monkeypatch):
    consumer = threading.get_ident()
    observed = []

    def prepare(value):
        observed.append(threading.get_ident())
        return value * 2

    def unexpected_pool(*args, **kwargs):
        pytest.fail("Synchronous calls must not construct a thread pool")

    monkeypatch.setattr(inference_input, "ThreadPoolExecutor", unexpected_pool)
    predictor = OrderedInputPredictor(prepare, lambda value: value + 1, prefetch_workers=0)
    assert list(predictor.iter_predictions(range(3))) == [1, 3, 5]
    predictor.prefetch_workers = 2
    assert predictor(5) == 11
    assert observed == [consumer] * 4


@pytest.mark.parametrize("failure", ["prepare", "infer", "source", "consumer"])
def test_failure_or_early_close_shuts_down_input_threads(failure):
    before = {thread.ident for thread in threading.enumerate()}

    def prepare(value):
        if failure == "prepare" and value == 1:
            raise RuntimeError("prepare failed")
        return value

    def infer(value):
        if failure == "infer" and value == 1:
            raise RuntimeError("infer failed")
        return value

    def paths():
        for value in range(20):
            if failure == "source" and value == 5:
                raise RuntimeError("source failed")
            yield value

    iterator = OrderedInputPredictor(prepare, infer).iter_predictions(paths())
    if failure == "consumer":
        assert next(iterator) == 0
        iterator.close()
    else:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            list(iterator)
    assert not [thread for thread in threading.enumerate()
                if thread.ident not in before and thread.name.startswith("baseline-input")]


@pytest.mark.parametrize("workers,bound", [(-1, 4), (1, 0)])
def test_invalid_input_pool_parameters_are_rejected(workers, bound):
    with pytest.raises(ValueError):
        OrderedInputPredictor(lambda value: value, lambda value: value, workers, bound)


def test_yolo_iteration_matches_single_image_preparation_and_predictions(tmp_path):
    from ultralytics.data.augment import LetterBox

    paths = make_images(tmp_path)
    consumer = threading.get_ident()
    seen_inputs = []
    generator = torch.Generator().manual_seed(17)
    protos = torch.randn(1, 2, 16, 16, generator=generator)
    outputs = torch.tensor([[[160., 320., 460.], [200., 320., 400.], [140., 160., 100.],
                             [180., 160., 130.], [.92, .83, .51], [1., -.3, .8], [.2, 1., -.7]]])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, tensor):
            assert threading.get_ident() == consumer
            assert tensor.shape == (1, 3, 640, 640) and tensor.dtype == torch.float32
            seen_inputs.append(tensor.clone())
            return outputs.clone(), protos

    model = Model()
    predictor = adapters.yolo_predictor(model, 1, COMMON_VALIDATION, 2)
    original_prepare = predictor.prepare
    prepare_threads = []

    def prepare(path):
        prepare_threads.append(threading.get_ident())
        tensor, shape = original_prepare(path)
        assert tensor.device.type == "cpu" and tensor.dtype == torch.uint8
        return tensor, shape

    predictor.prepare = prepare
    with torch.no_grad():
        expected = [predictor(path) for path in paths]
        prepare_threads.clear()
        actual = list(predictor.iter_predictions(paths))
    same_predictions(actual, expected)
    assert all(identity != consumer for identity in prepare_threads)
    for path, synchronous, prefetched in zip(paths, seen_inputs[:3], seen_inputs[3:]):
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        resized = LetterBox(new_shape=(640, 640), auto=False, stride=32)(image=np.ascontiguousarray(rgb[:, :, ::-1]))
        tensor = torch.from_numpy(np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1)))
        original = tensor.float().unsqueeze(0) / 255.0
        assert torch.equal(synchronous, original) and torch.equal(prefetched, original)


def test_rfdetr_iteration_keeps_normalize_resize_and_forward_in_consumer_thread(tmp_path, monkeypatch):
    from torchvision.transforms import functional as TF

    paths = make_images(tmp_path)
    consumer = threading.get_ident()
    seen_inputs = []
    generator = torch.Generator().manual_seed(29)
    outputs = dict(pred_logits=torch.randn(1, 7, 3, generator=generator),
                   pred_boxes=torch.randn(1, 7, 4, generator=generator),
                   pred_masks=torch.randn(1, 7, 5, 7, generator=generator))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, tensor):
            assert threading.get_ident() == consumer and tensor.shape[0] == 1
            seen_inputs.append(tensor.clone())
            return outputs

    original_normalize, original_resize = TF.normalize, TF.resize

    def normalize(*args, **kwargs):
        assert threading.get_ident() == consumer
        return original_normalize(*args, **kwargs)

    def resize(*args, **kwargs):
        assert threading.get_ident() == consumer
        return original_resize(*args, **kwargs)

    monkeypatch.setattr(TF, "normalize", normalize)
    monkeypatch.setattr(TF, "resize", resize)
    predictor = adapters.rfdetr_predictor(Model(), 24, 15, COMMON_VALIDATION, 8)
    assert predictor.prefetch_workers == predictor.max_prefetch == 2
    original_prepare = predictor.prepare
    prepare_threads = []

    def prepare(path):
        prepare_threads.append(threading.get_ident())
        tensor, width, height = original_prepare(path)
        assert tensor.device.type == "cpu" and tensor.shape == (3, height, width)
        return tensor, width, height

    predictor.prepare = prepare
    with torch.no_grad():
        expected = [predictor(path) for path in paths]
        prepare_threads.clear()
        actual = list(predictor.iter_predictions(paths))
    same_predictions(actual, expected)
    assert all(identity != consumer for identity in prepare_threads)
    for path, synchronous, prefetched in zip(paths, seen_inputs[:3], seen_inputs[3:]):
        with Image.open(path) as image:
            original = TF.to_tensor(image.convert("RGB"))
        original = original_resize(original_normalize(original, [.485, .456, .406], [.229, .224, .225]), (24, 24))
        assert torch.equal(synchronous[0], original) and torch.equal(prefetched[0], original)


def official_function(path, name, namespace):
    """Execute the pinned function body without installing its CUDA extension dependencies."""
    if not path.is_file():
        pytest.skip(f"Pinned official source fixture unavailable: {path.name}")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def mmdet_fixture(monkeypatch):
    """Use file-only CPU Compose and official pseudo_collate with a transparent detector."""
    deps = Path("E:/mastercode/_work/baseline_bug2_20260926/deps")
    collate_env = dict(Any=Any, Sequence=collections.abc.Sequence, Mapping=collections.abc.Mapping)
    pseudo_collate = official_function(deps / "mmengine/dataset/utils.py", "pseudo_collate", collate_env)
    pipelines, prepared_threads = [], []

    class RoIPool(torch.nn.Module):
        pass

    class Compose:
        def __init__(self, stages):
            pipelines.append(stages)

        def __call__(self, data):
            prepared_threads.append(threading.get_ident())
            assert set(data) == {"img_path", "img_id"} and data["img_id"] == 0
            with Image.open(data["img_path"]) as image:
                rgb = np.array(image.convert("RGB"))
            return dict(inputs=torch.from_numpy(rgb.transpose(2, 0, 1).copy()),
                        data_samples=types.SimpleNamespace(img_id=0, img_path=data["img_path"], ori_shape=rgb.shape[:2],
                                                           img_shape=rgb.shape[:2], scale_factor=(1., 1.)))

    def get_pipeline(cfg):
        return cfg["pipeline"]
    monkeypatch.setitem(sys.modules, "mmcv.ops", types.SimpleNamespace(RoIPool=RoIPool))
    monkeypatch.setitem(sys.modules, "mmdet.utils", types.SimpleNamespace(get_test_pipeline_cfg=get_pipeline))
    monkeypatch.setitem(sys.modules, "mmengine.dataset", types.SimpleNamespace(Compose=Compose,
                                                                            pseudo_collate=pseudo_collate))
    env = dict(torch=torch, np=np, nn=torch.nn, ImagesType=object, Optional=Optional, Union=Union,
               Compose=Compose, DetDataSample=object, SampleList=list, RoIPool=RoIPool,
               get_test_pipeline_cfg=get_pipeline)
    official_inference = official_function(deps / "mmdet/apis/inference.py", "inference_detector", env)
    return types.SimpleNamespace(Compose=Compose, RoIPool=RoIPool, pipelines=pipelines,
                                 prepared_threads=prepared_threads, official_inference=official_inference)


def test_mmdet_matches_official_single_image_pipeline_collation_and_outputs(tmp_path, mmdet_fixture):
    paths = make_images(tmp_path)
    consumer = threading.get_ident()
    seen_metadata = []
    cfg = dict(pipeline=[dict(type="LoadImageFromFile"), dict(type="Resize", scale=(640, 640), keep_ratio=True),
                         dict(type="mmdet.LoadAnnotations", with_bbox=True, with_mask=True),
                         dict(type="PackDetInputs", meta_keys=("img_id", "ori_shape", "img_shape", "scale_factor"))])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = cfg
            self.data_preprocessor = types.SimpleNamespace(device=torch.device("cpu"))

        def test_step(self, data):
            assert threading.get_ident() == consumer and not torch.is_grad_enabled()
            assert set(data) == {"inputs", "data_samples"}
            assert len(data["inputs"]) == len(data["data_samples"]) == 1
            tensor, sample = data["inputs"][0], data["data_samples"][0]
            seen_metadata.append(vars(sample).copy())
            instances = types.SimpleNamespace(masks=tensor[:1] > 70, scores=torch.tensor([.9]),
                                               labels=torch.tensor([0]))
            return [types.SimpleNamespace(pred_instances=instances)]

    model = Model()
    predictor = adapters.mmdet_predictor(model, cfg, 2)
    assert [stage["type"] for stage in mmdet_fixture.pipelines[0]] == ["LoadImageFromFile", "Resize", "PackDetInputs"]
    assert cfg["pipeline"][2]["type"] == "mmdet.LoadAnnotations"
    expected = []
    pipeline = mmdet_fixture.Compose(mmdet_fixture.pipelines[0])
    for path in paths:
        official = mmdet_fixture.official_inference(model, str(path), test_pipeline=pipeline).pred_instances
        expected.append((official.masks.numpy(), official.scores.numpy(), official.labels.numpy() + 1))
    synchronous = [predictor(path) for path in paths]
    mmdet_fixture.prepared_threads.clear()
    actual = list(predictor.iter_predictions(paths))
    same_predictions(actual, expected)
    same_predictions(synchronous, expected)
    assert all(identity != consumer for identity in mmdet_fixture.prepared_threads)
    assert seen_metadata[:3] == seen_metadata[3:6] == seen_metadata[6:]


def test_mmdet_requires_model_cfg_before_any_input(mmdet_fixture):
    model = torch.nn.Linear(1, 1)
    with pytest.raises(AttributeError, match="model.cfg"):
        adapters.mmdet_predictor(model, {})
    assert not mmdet_fixture.pipelines and not mmdet_fixture.prepared_threads


def test_mmdet_keeps_official_cpu_roipool_rejection(mmdet_fixture):
    model = torch.nn.Sequential(mmdet_fixture.RoIPool())
    model.cfg = {}
    model.data_preprocessor = types.SimpleNamespace(device=torch.device("cpu"))
    with pytest.raises(AssertionError, match="CPU inference with RoIPool"):
        adapters.mmdet_predictor(model, {})
    assert not mmdet_fixture.pipelines and not mmdet_fixture.prepared_threads


def test_unified_profile_uses_same_prefetched_predictor_and_preserves_results(tmp_path, monkeypatch, mmdet_fixture):
    import inference_runtime
    import mmdet_memory
    import validation

    images = tmp_path / "coco/images/val"
    images.mkdir(parents=True)
    paths = make_images(images)
    records = []
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            width, height = image.size
        records.append(dict(id=index + 1, file_name=path.name, width=width, height=height))
    annotation = tmp_path / "coco/annotations/instances_val.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(json.dumps(dict(images=records, annotations=[], categories=[dict(id=1, name="fruit")])))
    (tmp_path / "summary.json").write_text(json.dumps(dict(names=["fruit"])))
    output = tmp_path / "profile_output"
    output.mkdir()
    old_metrics = output / "val_common.json"
    old_metrics.write_text('{"previous": true}')
    seen_paths = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.cfg = dict(pipeline=[dict(type="LoadImageFromFile"), dict(type="PackDetInputs")])
            self.data_preprocessor = types.SimpleNamespace(device=torch.device("cpu"))

        def test_step(self, data):
            sample, tensor = data["data_samples"][0], data["inputs"][0]
            seen_paths.append(Path(sample.img_path).name)
            instances = types.SimpleNamespace(masks=tensor[:1] > 70, scores=torch.tensor([.9]),
                                               labels=torch.tensor([0]))
            return [types.SimpleNamespace(pred_instances=instances)]

    model = Model()
    monkeypatch.setitem(sys.modules, "mmdet.apis", types.SimpleNamespace(init_detector=lambda *args, **kwargs: model))
    monkeypatch.setattr(mmdet_memory, "install_memory_safe_rtmdet", lambda: {})
    monkeypatch.setattr(inference_runtime, "install_inference_runtime", lambda: {})
    monkeypatch.setattr(inference_runtime, "runtime_status", lambda: dict(adapter="test"))

    def unexpected_ap(*args, **kwargs):
        pytest.fail("Profile must not calculate AP or select a checkpoint")

    monkeypatch.setattr(validation, "evaluate_predictions", unexpected_ap)
    original_predictor = adapters.mmdet_predictor

    def predictor(candidate, cfg, workers):
        assert workers == 1
        return original_predictor(candidate, cfg, workers)

    monkeypatch.setattr(adapters, "mmdet_predictor", predictor)
    job = dict(seed=42, workers=1, recipe=dict(family="mmdet"), validation=COMMON_VALIDATION)
    metrics = adapters.evaluate_unified(job, tmp_path, tmp_path, tmp_path / "best.pth", output, profile_images=2)
    expected_order = [record["file_name"] for record in sorted(records,
                                                              key=lambda item: item["height"] * item["width"],
                                                              reverse=True)[:2]]
    assert seen_paths == expected_order
    assert metrics["profile_only"] is True and metrics["images"] == 2
    assert len(metrics["per_image_timing"]) == 2
    report = json.loads((output / "evaluation_timing.json").read_text())
    assert report["profile_only"] is True and report["runtime"]["adapter"] == "test"
    assert report["first_image_includes_warmup"] is True
    assert all("input_wait_seconds" in record for record in report["per_image_timing"])
    assert old_metrics.read_text() == '{"previous": true}'
    assert not (tmp_path / "validation_common/best.json").exists()
    assert not (output / "complete.json").exists()
