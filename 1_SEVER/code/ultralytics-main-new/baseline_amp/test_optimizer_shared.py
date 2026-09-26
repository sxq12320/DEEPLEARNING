"""Regression for bug2: RTMDet-style shared convolutions must be optimized exactly once."""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from mmdet_common import audit_optimizer_parameters, make_optim_wrapper_config
from baseline_amp.registry import make_queue


def test_non_yolo_queue_keeps_all_four_amp_pairs():
    queue = make_queue("non_yolo", [42], 300, 4)
    assert len(queue) == 8
    expected = {"rtmdet_ins_tiny", "mask_rcnn_r50", "solov2_light_r18", "rfdetr_seg_nano"}
    assert {job["model"] for job in queue} == expected
    for name in expected:
        pair = [job for job in queue if job["model"] == name]
        assert {job["amp"] for job in pair} == {False, True}
        assert pair[0]["recipe"] == pair[1]["recipe"]


def test_amp_modes_share_the_same_optimizer_recipe():
    plain, amp = make_optim_wrapper_config(False), make_optim_wrapper_config(True)
    assert plain["optimizer"] == amp["optimizer"]
    assert plain["paramwise_cfg"] == amp["paramwise_cfg"]
    assert plain["paramwise_cfg"]["bypass_duplicate"] is True
    assert amp["type"] == "AmpOptimWrapper" and amp["dtype"] == "float16"


def test_shared_convolution_reproduces_failure_and_fixed_optimizer_steps():
    torch = pytest.importorskip("torch")
    pytest.importorskip("mmengine")
    from mmengine.optim import build_optim_wrapper

    class SharedScales(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scales = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4)) for _ in range(3)
            ])
            for scale in self.scales[1:]:
                scale[0].weight = self.scales[0][0].weight
                scale[0].bias = self.scales[0][0].bias

        def forward(self, x):
            return sum(scale(x) for scale in self.scales)

    model = SharedScales()
    old = make_optim_wrapper_config(False)
    old["paramwise_cfg"].pop("bypass_duplicate")
    with pytest.raises(ValueError, match="more than one parameter group"):
        build_optim_wrapper(model, old)
    wrapper = build_optim_wrapper(model, make_optim_wrapper_config(False))
    report = audit_optimizer_parameters(model, wrapper)
    assert report["parameter_tensors"] == len(list(model.parameters()))
    assert report["duplicate_tensors"] == report["missing_trainable_tensors"] == 0
    by_id = {id(p): group for group in wrapper.optimizer.param_groups for p in group["params"]}
    for scale in model.scales:
        assert by_id[id(scale[0].weight)]["weight_decay"] == 0.0005
        assert by_id[id(scale[0].bias)]["weight_decay"] == 0
        assert by_id[id(scale[1].weight)]["weight_decay"] == 0
    before = model.scales[0][0].weight.detach().clone()
    wrapper.update_params(model(torch.randn(2, 3, 8, 8)).square().mean())
    assert not torch.equal(before, model.scales[0][0].weight)


def test_cli_can_skip_yolo_without_replacing_server_settings(monkeypatch, capsys, tmp_path):
    from baseline_amp.batch import main

    monkeypatch.setattr(sys, "argv", ["runner", "--dry-run", "--suite", "non_yolo", "--project", str(tmp_path)])
    settings = dict(SUITE="all", DATA="server-owned-data.yaml", PROJECT="old-project", DEVICE=1,
                    SEEDS=[42], EPOCHS=300, WORKERS=4, BATCHES={}, AMP_MODES=[1, 0])
    main(settings)
    output = capsys.readouterr().out
    assert "8 jobs" in output and "rtmdet_ins_tiny_amp" in output
    assert "yolo11n_seg_amp" not in output
    assert settings["SUITE"] == "all" and settings["PROJECT"] == "old-project"
