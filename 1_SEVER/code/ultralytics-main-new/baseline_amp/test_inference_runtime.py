"""Numerical/dispatch/provenance tests; CUDA-only checks also run in server preflight."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inference_runtime as runtime


@pytest.fixture
def installed(monkeypatch):
    original = functional.interpolate
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setattr(functional, "interpolate", original)
    monkeypatch.setattr(runtime, "_ORIGINAL_INTERPOLATE", None)
    monkeypatch.setattr(runtime, "_NATIVE_CALLS", 0)
    runtime.install_inference_runtime()
    yield original
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.mark.parametrize(
    "options",
    [
        dict(size=21),
        dict(size=(23, 17)),
        dict(scale_factor=2),
        dict(scale_factor=(1.7, 2.3)),
        dict(scale_factor=(1.7, 2.3), recompute_scale_factor=True),
    ],
)
@pytest.mark.parametrize("align_corners", [False, True, None])
def test_native_forward_matches_stock_coordinate_conventions(options, align_corners):
    value = torch.randn(2, 3, 7, 9, generator=torch.Generator().manual_seed(8))
    expected = functional.interpolate(value, mode="bilinear", align_corners=align_corners, **options)
    actual = runtime._native_bilinear(
        value, options.get("size"), options.get("scale_factor"), align_corners, options.get("recompute_scale_factor")
    )
    assert torch.equal(actual, expected)


def test_training_dispatch_and_gradients_unchanged(installed, monkeypatch):
    def forbidden(*args):
        raise AssertionError("Native inference bypass must never run in training")

    monkeypatch.setattr(runtime, "_native_bilinear", forbidden)
    value = torch.randn(1, 2, 5, 7, requires_grad=True)
    actual = functional.interpolate(value, size=(13, 17), mode="bilinear", align_corners=False)
    expected = installed(value, size=(13, 17), mode="bilinear", align_corners=False)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual.sum(), value)[0], torch.autograd.grad(expected.sum(), value)[0])


def test_bypass_only_no_grad_cuda_deterministic_bilinear(installed, monkeypatch):
    fake = SimpleNamespace(is_cuda=True, ndim=4)
    called = []
    monkeypatch.setattr(runtime, "_native_bilinear", lambda *args: called.append(args) or "native")
    monkeypatch.setattr(runtime, "_ORIGINAL_INTERPOLATE", lambda *args, **kwargs: "original")
    torch.use_deterministic_algorithms(True, warn_only=True)
    assert functional.interpolate(fake, size=(5, 7), mode="bilinear") == "original"
    with torch.no_grad(), runtime.inference_runtime_scope():
        assert functional.interpolate(fake, size=(5, 7), mode="bilinear") == "native"
        assert functional.interpolate(fake, size=(5, 7), mode="nearest") == "original"
        assert functional.interpolate(fake, size=(5, 7), mode="bilinear", antialias=True) == "original"
        torch.use_deterministic_algorithms(False)
        assert functional.interpolate(fake, size=(5, 7), mode="bilinear") == "original"
    assert len(called) == 1


@pytest.fixture
def fake_cuda_dispatch(installed, monkeypatch):
    fake = SimpleNamespace(is_cuda=True, ndim=4)
    monkeypatch.setattr(runtime, "_native_bilinear", lambda *args: "native")
    monkeypatch.setattr(runtime, "_ORIGINAL_INTERPOLATE", lambda *args, **kwargs: "original")
    torch.use_deterministic_algorithms(True)
    return lambda: functional.interpolate(fake, size=(5, 7), mode="bilinear")


def test_training_no_grad_without_explicit_scope_keeps_original_dispatch(fake_cuda_dispatch):
    with torch.no_grad():
        assert fake_cuda_dispatch() == "original"
    assert runtime.runtime_status()["explicit_inference_scope_required"] is True


def test_explicit_scope_does_not_override_enabled_gradients(fake_cuda_dispatch):
    with torch.enable_grad(), runtime.inference_runtime_scope():
        assert fake_cuda_dispatch() == "original"
    with torch.no_grad(), runtime.inference_runtime_scope():
        assert fake_cuda_dispatch() == "native"


def test_inference_scope_nests_and_restores_after_exceptions(fake_cuda_dispatch):
    with torch.no_grad():
        assert fake_cuda_dispatch() == "original"
        with pytest.raises(RuntimeError, match="outer failure"):
            with runtime.inference_runtime_scope():
                assert fake_cuda_dispatch() == "native"
                with pytest.raises(ValueError, match="inner failure"):
                    with runtime.inference_runtime_scope():
                        assert fake_cuda_dispatch() == "native"
                        raise ValueError("inner failure")
                assert fake_cuda_dispatch() == "native"
                raise RuntimeError("outer failure")
        assert fake_cuda_dispatch() == "original"
        with runtime.inference_runtime_scope():
            assert fake_cuda_dispatch() == "native"
        assert fake_cuda_dispatch() == "original"


def test_inference_scope_does_not_leak_to_input_threads(fake_cuda_dispatch):
    def prepare_in_worker():
        with torch.no_grad():
            before = fake_cuda_dispatch()
            with runtime.inference_runtime_scope():
                inside = fake_cuda_dispatch()
            return before, inside, fake_cuda_dispatch()

    with torch.no_grad(), runtime.inference_runtime_scope():
        assert fake_cuda_dispatch() == "native"
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(prepare_in_worker).result() == ("original", "native", "original")
        assert fake_cuda_dispatch() == "native"
    with torch.no_grad():
        assert fake_cuda_dispatch() == "original"


@pytest.mark.parametrize("failure", ["none", "before_sync", "save"])
def test_legacy_validation_hook_closes_scope_and_supports_cpu(tmp_path, monkeypatch, failure):
    pytest.importorskip("mmengine")
    import common

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def unexpected_sync():
        pytest.fail("CPU validation must not synchronize CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_sync)
    if failure == "save":
        def fail_save(*args):
            raise RuntimeError("save failed")

        monkeypatch.setattr(common, "save_json", fail_save)
    hook = runtime.make_validation_timing_hook(tmp_path)
    runner = SimpleNamespace(epoch=3, logger=SimpleNamespace(info=lambda message: None))
    if failure == "before_sync":
        monkeypatch.setattr(hook, "_synchronize", lambda: (_ for _ in ()).throw(RuntimeError("sync failed")))
        with pytest.raises(RuntimeError, match="sync failed"):
            hook.before_val_epoch(runner)
    else:
        hook.before_val_epoch(runner)
        assert runtime._INFERENCE_SCOPE_DEPTH.get() == 1
        hook.before_val_iter(runner, 0)
        hook.after_val_iter(runner, 0)
        if failure == "save":
            with pytest.raises(RuntimeError, match="save failed"):
                hook.after_val_epoch(runner)
        else:
            hook.after_val_epoch(runner)
            assert (tmp_path / "validation_timing.json").is_file()
        hook.after_run(runner)
    assert hook._inference_scope is None
    assert runtime._INFERENCE_SCOPE_DEPTH.get() == 0


def test_invalid_arguments_delegate_to_stock_errors(installed):
    value = torch.randn(1, 2, 5, 7)
    assert runtime._native_bilinear(value, (3, 4), 2, False, None) is None
    assert runtime._native_bilinear(value, (3, 4), None, False, True) is None
    assert runtime._native_bilinear(value, None, (-1, 2), False, None) is None
    with pytest.raises(ValueError):
        functional.interpolate(value, size=(3, 4), scale_factor=2, mode="bilinear")


def test_install_is_idempotent_and_flags_restored(installed):
    torch.use_deterministic_algorithms(True, warn_only=True)
    runtime.install_inference_runtime()
    runtime.install_inference_runtime()
    with torch.no_grad():
        result = functional.interpolate(torch.ones(1, 1, 3, 4), size=8, mode="bilinear")
    assert result.shape == (1, 1, 8, 8)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Server CUDA required; local runtime is CPU-only")
def test_cuda_native_vs_deterministic_decomposition(installed):
    torch.use_deterministic_algorithms(True)
    value = torch.randn(1, 4, 7, 9, device="cuda")
    with torch.no_grad():
        expected = installed(value, size=(81, 103), mode="bilinear", align_corners=False)
        with runtime.inference_runtime_scope():
            actual = functional.interpolate(value, size=(81, 103), mode="bilinear", align_corners=False)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert runtime.runtime_status()["native_bilinear_calls"] > 0
    assert torch.are_deterministic_algorithms_enabled()


@pytest.mark.parametrize("case", ["empty", "one_pixel", "full", "random", "probabilities"])
def test_rle_bbox_matches_previous_mask_scan(case):
    from coco_utils import mask_bbox, prediction_from_mask

    mask = np.zeros((19, 27), dtype=bool)
    if case == "one_pixel":
        mask[4, 7] = True
    elif case == "full":
        mask[:] = True
    elif case == "random":
        mask = np.random.default_rng(2).uniform(size=mask.shape) > 0.6
    elif case == "probabilities":
        mask = np.random.default_rng(2).uniform(size=mask.shape)
    prediction = prediction_from_mask(1, 1, 0.9, mask)
    assert prediction["bbox"] == mask_bbox(mask > 0.5)


def test_validation_io_only_removes_unused_gt_masks():
    pytest.importorskip("mmengine")
    from mmengine.config import ConfigDict
    from mmdet_common import configure_validation_io

    cfg = ConfigDict()
    for split in ("val", "test"):
        cfg[f"{split}_evaluator"] = dict(type="CocoMetric", ann_file="original.json")
        cfg[f"{split}_dataloader"] = dict(
            batch_size=1,
            dataset=dict(
                type="CocoDataset",
                pipeline=[
                    dict(type="Resize", scale=(640, 640), keep_ratio=True),
                    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
                ],
            ),
        )
    configure_validation_io(cfg)
    assert cfg.val_dataloader.dataset.pipeline[0].scale == (640, 640)
    assert cfg.val_evaluator.ann_file == "original.json"
    assert cfg.val_dataloader.dataset.pipeline[1].with_mask is False
    assert cfg.val_dataloader.dataset.pipeline[1].with_bbox is True
    cfg.val_evaluator.ann_file = None
    with pytest.raises(ValueError, match="explicit ann_file"):
        configure_validation_io(cfg)


def test_worker_training_exits_before_evaluation(tmp_path, monkeypatch):
    import worker

    run_dir = tmp_path / "run"
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "summary.json").write_text(json.dumps(dict(names=["fruit"])))
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    job = dict(model="fake", recipe=dict(family="yolo"), run_dir=str(run_dir), prepared=str(prepared))
    job_file = tmp_path / "job.json"
    job_file.write_text(json.dumps(job))
    monkeypatch.setattr(worker, "preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "snapshot", lambda: {})
    monkeypatch.setattr(worker, "train_yolo", lambda *args: checkpoint)
    monkeypatch.setattr(worker, "evaluate", lambda *args, **kwargs: pytest.fail("Trainer must exit first"))
    monkeypatch.setattr(sys, "argv", ["worker.py", "--job", str(job_file), "--stage", "train"])
    worker.main()
    assert (run_dir / "trained.json").is_file()
    assert not (run_dir / "complete.json").exists()


def test_worker_profile_does_not_mark_complete(tmp_path, monkeypatch):
    import worker

    job = dict(model="fake", recipe=dict(family="yolo"), run_dir=str(tmp_path), prepared=str(tmp_path))
    job_file = tmp_path / "job.json"
    job_file.write_text(json.dumps(job))
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    output = tmp_path / "profile"
    monkeypatch.setattr(worker, "preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "evaluate", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worker.py",
            "--job",
            str(job_file),
            "--stage",
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--profile-images",
            "5",
        ],
    )
    worker.main()
    assert not (tmp_path / "complete.json").exists()
    assert output.is_dir()


def test_yolo_original_masks_bounded_on_cpu():
    from ultralytics.utils import ops

    value = torch.randn(8, 12, 12, generator=torch.Generator().manual_seed(7))
    coefficients = torch.randn(25, 8, generator=torch.Generator().manual_seed(9))
    boxes = torch.tensor([[2.0, 3.0, 29.0, 31.0]]).repeat(25, 1)
    expected = ops.process_mask_native(value, coefficients, boxes, (36, 40))
    actual = runtime.decode_yolo_masks_bounded(
        ops.process_mask_native, value, coefficients, boxes, (36, 40), pixel_budget=36 * 40 * 3
    )
    assert actual.device.type == "cpu" and torch.equal(actual, expected)
    assert runtime.decode_yolo_masks_bounded(
        ops.process_mask_native, value, coefficients[:0], boxes[:0], (36, 40)
    ).shape == (0, 36, 40)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Server CUDA required; local runtime is CPU-only"))])
@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8])
@pytest.mark.parametrize("count", [0, 1, 7])
def test_yolo_mask_transfer_preserves_empty_dtype_and_chunk_order(monkeypatch, device, dtype, count):
    shape = (7, 11)
    # A transposed mask also exercises a strided copy directly into final CPU storage.
    masks = (torch.arange(count * 77, device=device).reshape(count, 11, 7) % 3 == 0).transpose(1, 2).to(dtype)
    expected = masks.to(device="cpu", dtype=torch.uint8)
    coefficients = torch.arange(count, device=device).reshape(count, 1)
    boxes = torch.zeros(count, 4, device=device)
    chunks = []

    def decode(protos, selected, selected_boxes, output_shape):
        assert output_shape == shape and len(selected_boxes) == len(selected)
        chunks.append(len(selected))
        return masks[selected[:, 0]]

    def forbidden_cpu(*args, **kwargs):
        pytest.fail("Decoded masks must copy directly into final CPU storage without a CPU temporary")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden_cpu)
    actual = runtime.decode_yolo_masks_bounded(decode, None, coefficients, boxes, shape, pixel_budget=77 * 2)
    assert actual.device.type == "cpu" and actual.dtype == torch.uint8
    assert not actual.is_pinned() and actual.shape == (count, *shape)
    assert torch.equal(actual, expected)
    assert chunks == ([2, 2, 2, 1] if count == 7 else [1] if count == 1 else [])


def test_recovery_latest_checkpoint_marker(tmp_path):
    from baseline_amp.recovery import find_checkpoint

    train = tmp_path / "train"
    train.mkdir()
    checkpoint = train / "epoch_48.pth"
    checkpoint.touch()
    (train / "last_checkpoint").write_text("epoch_48.pth")
    assert find_checkpoint(tmp_path) == checkpoint.resolve()
    selected = train / "best.pth"
    selected.touch()
    (tmp_path / "trained.json").write_text(json.dumps(dict(checkpoint=str(selected))))
    assert find_checkpoint(tmp_path) == selected.resolve()


def test_recovery_dry_run_leaves_existing_experiment_unchanged(tmp_path, monkeypatch, capsys):
    from baseline_amp.recovery import main

    run = tmp_path / "model"
    run.mkdir()
    job = dict(run_dir=str(run), seed=42, recipe=dict(family="mmdet", environment="mmdet"))
    (run / "job.json").write_text(json.dumps(job))
    (run / "train").mkdir()
    checkpoint = run / "train/epoch_2.pth"
    checkpoint.touch()
    (run / "train/last_checkpoint").write_text(str(checkpoint))
    settings = dict(
        RUN_DIR=str(run), MODE="resume", PROFILE_IMAGES=5, CHECKPOINT="", DEVICE=1, PYTHONS=dict(mmdet=sys.executable)
    )
    monkeypatch.setattr(sys, "argv", ["recovery", "--dry-run"])
    main(settings)
    output = capsys.readouterr().out
    assert "--resume-train" in output and "evaluate" in output
    assert not list(run.glob("runtime_*"))
    assert json.loads((run / "job.json").read_text()) == job


def test_resume_uses_saved_training_recipe_and_preserves_initial_hash(tmp_path, monkeypatch):
    pytest.importorskip("mmengine")
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.optim import build_optim_wrapper
    import worker
    import mmdet_memory

    original = dict(
        train_cfg=dict(type="EpochBasedTrainLoop", max_epochs=300, val_interval=1),
        train_dataloader=dict(dataset=dict(pipeline=[dict(type="LoadAnnotations", with_mask=True)])),
        randomness=dict(seed=42, deterministic=True),
        optim_wrapper=dict(type="OptimWrapper", optimizer=dict(type="AdamW", lr=0.003, weight_decay=0.0007)),
    )
    for split in ("val", "test"):
        original[f"{split}_evaluator"] = dict(type="CocoMetric", ann_file="same_annotations.json")
        original[f"{split}_dataloader"] = dict(
            batch_size=1, dataset=dict(pipeline=[dict(type="LoadAnnotations", with_mask=True)])
        )
    cfg_path = tmp_path / "effective_config.py"
    Config(original).dump(str(cfg_path))
    before = cfg_path.read_bytes()
    initial = tmp_path / "initialization.json"
    initial.write_text(json.dumps(dict(sha256="original_initialized_weights")))
    (tmp_path / "train").mkdir()
    checkpoint = tmp_path / "train/epoch_48.pth"
    checkpoint.touch()
    (tmp_path / "train/last_checkpoint").write_text(str(checkpoint))
    best = tmp_path / "train/best_coco_segm_mAP_epoch_47.pth"
    best.touch()
    captured = {}

    class FakeRunner:
        def __init__(self, cfg):
            captured["cfg"] = cfg
            self.model = torch.nn.Linear(2, 1)
            self.optim_wrapper = build_optim_wrapper(self.model, cfg.optim_wrapper)
            self.hooks = []

        def register_hook(self, hook, priority):
            self.hooks.append(hook)

        def train(self):
            for hook in self.hooks:
                hook.before_train(self)

    monkeypatch.setattr(Runner, "from_cfg", lambda cfg: FakeRunner(cfg))
    monkeypatch.setattr(mmdet_memory, "install_memory_safe_rtmdet", lambda: {})
    assert worker.train_mmdet(dict(seed=42, amp=False), tmp_path, tmp_path, resume=True) == best
    cfg = captured["cfg"]
    assert cfg.resume is True and cfg.load_from == str(checkpoint)
    assert cfg.optim_wrapper.optimizer.lr == 0.003 and cfg.optim_wrapper.optimizer.weight_decay == 0.0007
    assert cfg.train_cfg.max_epochs == 300 and cfg.train_cfg.val_interval == 1
    assert cfg.train_dataloader.dataset.pipeline[0].with_mask is True
    assert cfg.val_dataloader.dataset.pipeline[0].with_mask is False
    assert cfg_path.read_bytes() == before
    assert json.loads(initial.read_text())["sha256"] == "original_initialized_weights"


def test_official_coco_metric_process_needs_no_gt_bitmaps_with_ann_file():
    """Execute the unmodified official process method, not a custom substitute metric."""
    import ast
    import importlib.util
    from copy import deepcopy
    from pycocotools import mask as mask_utils

    spec = importlib.util.find_spec("mmdet")
    if spec is None:
        pytest.skip("Pinned source not installed")
    path = Path(spec.origin).parent / "evaluation/metrics/coco_metric.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CocoMetric")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "process")
    method.decorator_list = []
    for argument in method.args.args:
        argument.annotation = None
    method.returns = None
    namespace = dict(
        torch=torch,
        encode_mask_results=lambda masks: [
            mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in masks
        ],
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), namespace)
    prediction = dict(
        bboxes=torch.tensor([[1.0, 1.0, 3.0, 3.0]]),
        scores=torch.tensor([0.9]),
        labels=torch.tensor([0]),
        masks=torch.ones(1, 5, 7, dtype=torch.bool),
    )
    data = dict(img_id=1, ori_shape=(5, 7), pred_instances=prediction)
    with_gt = deepcopy(data)
    with_gt["gt_instances"] = dict(masks=torch.ones(1, 5, 7))
    metric = SimpleNamespace(_coco_api=object(), results=[])
    namespace["process"](metric, {}, [data, with_gt])
    assert metric.results[0][0] == metric.results[1][0]
    for key, value in metric.results[0][1].items():
        other = metric.results[1][1][key]
        assert np.array_equal(value, other) if isinstance(value, np.ndarray) else value == other


def test_profile_evaluate_selects_large_photos_without_overwriting_metrics(tmp_path, monkeypatch, installed):
    import worker
    import mmdet_memory
    import validation_adapters

    annotation = tmp_path / "coco/annotations/instances_val.json"
    annotation.parent.mkdir(parents=True)
    records = [
        dict(id=1, file_name="small.jpg", height=5, width=7),
        dict(id=2, file_name="large.jpg", height=11, width=13),
    ]
    annotation.write_text(json.dumps(dict(images=records)))
    (tmp_path / "summary.json").write_text(json.dumps(dict(names=["fruit"])))
    old_metrics = tmp_path / "val_common.json"
    old_metrics.write_text('{"saved_old_result": true}')
    model = torch.nn.Linear(2, 1)
    model.cfg = dict(randomness=dict(deterministic=True))
    paths = []

    def predict(model, path):
        paths.append(path)
        assert torch.are_deterministic_algorithms_enabled()
        instances = SimpleNamespace(
            masks=torch.ones(1, 11, 13, dtype=torch.bool), scores=torch.tensor([0.9]), labels=torch.tensor([0])
        )
        return SimpleNamespace(pred_instances=instances)

    monkeypatch.setitem(
        sys.modules,
        "mmdet.apis",
        SimpleNamespace(init_detector=lambda *args, **kwargs: model, inference_detector=predict),
    )
    monkeypatch.setattr(mmdet_memory, "install_memory_safe_rtmdet", lambda: {})
    def make_predictor(candidate, cfg):
        def file_predict(path):
            result = predict(candidate, str(path)).pred_instances
            return result.masks.numpy(), result.scores.numpy(), result.labels.numpy() + 1
        return file_predict
    monkeypatch.setattr(validation_adapters, "mmdet_predictor", make_predictor)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    job = dict(seed=42, recipe=dict(family="mmdet"))
    output = tmp_path / "profile_output"
    assert worker.evaluate(job, tmp_path, tmp_path, tmp_path / "best.pth", profile_images=1, output_dir=output) is None
    assert len(paths) == 1 and Path(paths[0]).name == "large.jpg"
    report = json.loads((output / "evaluation_timing.json").read_text())
    assert report["profile_only"] is True and report["images"][0]["instances"] == 1
    assert old_metrics.read_text() == '{"saved_old_result": true}'
    assert not (output / "val_common.json").exists()
