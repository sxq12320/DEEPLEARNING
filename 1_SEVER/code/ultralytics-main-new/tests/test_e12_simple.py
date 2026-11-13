"""Verify controlled removal, standard loss, prediction contract and cost."""
from copy import deepcopy
import torch
import pytest
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops
from ultralytics.utils.loss import v8SegmentationLoss
from citrus_paper_suite import ROOT
from ultralytics.nn.modules import SegmentCitrusE12Simple


@pytest.mark.parametrize("name", ["ES00_baseline", "ES01_detail", "ES02_semantic", "ES03_detail_semantic"])
def test_standard_loss_build_backward_and_grid(name):
    torch.set_num_threads(2)
    model = SegmentationModel(str(ROOT / "0_orange_yaml/E12_SIMPLE" / (name + ".yaml")), verbose=False)
    model.args = IterableSimpleNamespace(**dict(DEFAULT_CFG_DICT, mask_ratio=4, nwd_ratio=0.0))
    assert type(model.init_criterion()) is v8SegmentationLoss
    masks = torch.zeros(2, 32, 32)
    masks[:, 8:24, 8:24] = 1
    batch = dict(img=torch.rand(2, 3, 128, 128), masks=masks, batch_idx=torch.tensor([0., 1.]),
                 cls=torch.zeros(2, 1), bboxes=torch.tensor([[.5,.5,.5,.5]]*2))
    loss, _ = model.loss(batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    head = model.model[-1]
    assert hasattr(head, "detail_residual") == (name in ("ES01_detail", "ES03_detail_semantic"))
    assert hasattr(head, "semantic") == (name in ("ES02_semantic", "ES03_detail_semantic"))
    assert not hasattr(head, "quality_predictor")
    assert head.stride.tolist() == [8., 16., 32.]
    model.eval()
    with torch.no_grad():
        pred = model(batch["img"])
    assert pred[0][1].shape[-2:] == (32, 32)
    cost = get_flops(model, 640)
    assert 0 < cost < 12
    print(name, sum(p.numel() for p in model.parameters()), cost)


def test_zero_start_matches_standard_head_and_both_paths_can_learn():
    torch.set_num_threads(2)
    from ultralytics.nn.modules import Segment
    base = Segment(1, 32, 64, ch=(64, 128, 256)).eval()
    head = SegmentCitrusE12Simple(1, 32, 64, True, True, ch=(64, 128, 256, 64)).eval()
    head.load_state_dict(base.state_dict(), strict=False)
    base.stride = head.stride = torch.tensor([8., 16., 32.])
    x = [torch.randn(2, c, s, s) for c, s in [(64,16), (128,8), (256,4), (64,32)]]
    with torch.no_grad():
        a, b = base(x[:3])[0], head(x)[0]
    torch.testing.assert_close(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])
    head.train()
    pred = head(x)
    (pred['proto'].square().mean() + pred['scores'].square().mean()).backward()
    assert head.detail_residual[-1].weight.grad.abs().sum() > 0
    assert head.semantic[0].gain.grad.abs().sum() > 0
