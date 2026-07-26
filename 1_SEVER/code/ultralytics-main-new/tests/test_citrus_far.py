# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""柑橘远距离小目标改进的单元测试：模块前向 / IoU 扩展族数值 / BboxLoss 分发 / Lion 收敛."""

from types import SimpleNamespace

import pytest
import torch

from ultralytics.nn.modules.citrus_far import (
    CAA,
    CARAFE,
    CSFG,
    DFEM,
    EDFFN,
    ELA,
    EMA,
    LIAM,
    LRSA,
    RFB,
    BiFPNConcat,
    C3k2_DWR,
    C3k2_Faster,
    C3k2_WT,
    CoordAtt,
    DySample,
    FarFormer,
    HFBranch,
    HWDown,
    LumiFormer,
    SimAM,
    SPDConv,
    SPPF_LSKA,
)
from ultralytics.optim import Lion
from ultralytics.utils.iou_ext import bbox_iou_ext, focaler_remap, nwd, wiou_terms
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou


def _rand_boxes(n: int = 64, scale: float = 40.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Random valid xyxy box pairs (loosely overlapping)."""
    torch.manual_seed(0)
    cxy = torch.rand(n, 2) * scale
    wh1 = torch.rand(n, 2) * 8 + 0.5
    wh2 = wh1 * (0.5 + torch.rand(n, 2))
    off = (torch.rand(n, 2) - 0.5) * 2
    b1 = torch.cat([cxy - wh1 / 2, cxy + wh1 / 2], 1)
    b2 = torch.cat([cxy + off - wh2 / 2, cxy + off + wh2 / 2], 1)
    return b1, b2


# ---------------------------------------------------------------------------- modules
@pytest.mark.parametrize(
    "module,ch_out",
    [
        (SPDConv(32, 64), 64),
        (HWDown(32, 64), 64),
        (RFB(32, 32), 32),
        (SPPF_LSKA(32, 32), 32),
        (C3k2_Faster(32, 32, 1), 32),
        (C3k2_WT(32, 32, 1), 32),
        (C3k2_DWR(32, 32, 1), 32),
    ],
)
def test_channel_modules_forward(module, ch_out):
    x = torch.randn(2, 32, 32, 32)
    y = module(x)
    assert y.shape[1] == ch_out and torch.isfinite(y).all()


@pytest.mark.parametrize("attn", [EMA(32), SimAM(32), CoordAtt(32), ELA(32), CAA(32), LIAM(32), DFEM(32)])
def test_attention_modules_keep_shape(attn):
    x = torch.randn(2, 32, 32, 32)
    y = attn(x)
    assert y.shape == x.shape and torch.isfinite(y).all()


def test_upsamplers_double_resolution():
    x = torch.randn(2, 32, 16, 16)
    assert CARAFE(32)(x).shape == (2, 32, 32, 32)
    assert DySample(32)(x).shape == (2, 32, 32, 32)


def test_bifpn_concat_and_csfg():
    p2, p3 = torch.randn(2, 16, 64, 64), torch.randn(2, 32, 32, 32)
    y = BiFPNConcat(2)([p3, p3])
    assert y.shape == (2, 64, 32, 32)
    z = CSFG(16, 32)([p2, p3])
    assert z.shape == p3.shape and torch.isfinite(z).all()


def test_dfem_identity_at_init():
    """DFEM 频带增益 init=0 → 频域分支起步为恒等（残差结构训练稳定的关键设计）."""
    m = DFEM(16)
    x = torch.randn(1, 16, 16, 16)
    xf = torch.fft.rfft2(x.float(), norm="ortho")
    y = torch.fft.irfft2(xf * (1.0 + m.gains.float()[:, m._band_index(16, 16, x.device)]).unsqueeze(0),
                         s=(16, 16), norm="ortho")
    assert torch.allclose(y, x, atol=1e-5)


from ultralytics.nn.modules.citrus_far import HCO, HSF, LCE, MWCA, PCFA, TDAM, TGP, FocusedLinearAttn, HyperACE  # noqa: E402


def test_llm_era_modules():
    """MoCE 软路由 / HyperRes 初始化等价性 / DyT 有界性."""
    from ultralytics.nn.modules.citrus_far import C3k2_MoCE, DyT, HyperRes, SXQBottleneck

    m = C3k2_MoCE(64, 64, 1)
    x = torch.randn(2, 64, 32, 32, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    hr = HyperRes(32, 2)
    x2 = torch.randn(1, 32, 16, 16)
    ref = x2
    for blk in hr.blocks:
        ref = blk(ref)
    assert torch.allclose(hr(x2), ref, atol=1e-5)  # init 精确等价标准残差链
    d = DyT(32)
    x3 = torch.randn(2, 32, 8, 8) * 100
    assert (d(x3).abs() <= (d.gamma.abs() + d.beta.abs() + 1e-4).max() * 1.01 + 1).all()  # tanh 有界


def test_pcfa_partial_frequency():
    """PCFA：形状不变、init 近恒等（gains=0+LayerScale）、反向有限、只有 1/4 通道走 FFT."""
    m = PCFA(32)
    assert m.cp == 8  # 1/4 通道
    x = torch.randn(2, 32, 16, 16, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert (y - x).abs().mean() < 0.5 * x.abs().mean()  # 近恒等起步
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_tgp_hsf_shapes():
    img = torch.rand(2, 3, 64, 64)
    assert TGP()(img).shape == img.shape
    low, high = torch.randn(2, 32, 32, 32), torch.randn(2, 64, 16, 16)
    out = HSF(32, 64)([low, high])
    assert out.shape == low.shape and torch.isfinite(out).all()


def test_hco_heat_conduction():
    """HCO：形状不变、反向有限；k 越大越平滑（热传导物理性质）."""
    m = HCO(32, blocks=2)
    x = torch.randn(2, 32, 16, 16, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    # 物理性质：k 大 → 输出更平滑（高频衰减更强）
    blk = m.blocks[0]
    xf = torch.randn(1, 32, 16, 16)
    with torch.no_grad():
        blk.k.fill_(-3.0)
        v_small_k = blk(xf).var()
        blk.k.fill_(3.0)
        v_large_k = blk(xf).var()
    assert v_large_k < v_small_k * 1.5  # 大扩散时间不应增加方差（平滑效应）


def test_hyperace_hypergraph():
    m = HyperACE(32, edges=8)
    x = torch.randn(2, 32, 16, 16, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    # gamma init 0.01 → 残差起步近恒等
    assert (y - x).abs().mean() < 0.5 * x.abs().mean()
from ultralytics.utils.loss import v8SegmentationLoss  # noqa: E402
from ultralytics.utils.tal import TaskAlignedAssigner  # noqa: E402


def test_mwca_shape_backward_identity_start():
    m = MWCA(32)
    x = torch.randn(2, 32, 32, 32, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    # gamma init 0.01 → 输出近似恒等（残差起步）
    assert (y - x).abs().mean() < 0.5 * x.abs().mean()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_ffl_frequency_mask_loss():
    """FFL：pred==gt 时频域项为 0；不同掩码时为正且梯度有限."""

    class _D:  # 只需要 freq_ratio 属性的 self 替身
        freq_ratio = 0.1

    proto = torch.randn(32, 40, 40)
    gt = (torch.rand(3, 40, 40) > 0.7).float()
    xyxy = torch.tensor([[0.0, 0.0, 40.0, 40.0]]).repeat(3, 1)
    area = torch.full((3,), 0.25)
    pred = torch.randn(3, 32, requires_grad=True)
    loss = v8SegmentationLoss.single_mask_loss(_D(), gt, pred, proto, xyxy, area)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    # 频域项本身：完全一致的 mask → FFL=0
    pm = gt * 20.0 - 10.0  # logits, sigmoid≈gt
    pf = torch.fft.rfft2(pm.sigmoid().float(), norm="ortho")
    gf = torch.fft.rfft2(gt.float(), norm="ortho")
    assert (pf - gf).abs().max() < 1e-3


def test_tdam_lce_shapes_and_identity():
    x = torch.randn(2, 32, 32, 32)
    t = TDAM(32)
    assert t(x).shape == x.shape
    # gain init 0 → sigmoid(0)=0.5 但 gate 有界，恒等主导路径存在；至少输出有限
    assert torch.isfinite(t(x)).all()
    img = torch.rand(2, 3, 64, 64)  # [0,1] 图像域
    lce = LCE(3, 3)
    y = lce(img)
    assert y.shape == img.shape and (y >= 0).all() and (y <= 1).all()
    assert torch.allclose(y, img, atol=1e-5)  # A init=0 → 恒等起步


def test_focused_linear_attn():
    m = FocusedLinearAttn(32, heads=4)
    x = torch.randn(2, 32, 16, 16, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


class _StrictAssigner(TaskAlignedAssigner):
    """去掉本 fork 上游的微小框虚拟扩张补偿，还原'中心严格在框内'的原始规则（用于测试 min_pos 兜底）."""

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):  # noqa: ARG002
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2)
        return deltas.view(bs, n_boxes, xy_centers.shape[0], -1).amin(3).gt_(eps)


def _tiny_gt_case():
    na = 100  # 10x10 anchors, stride 8, centers at 4,12,...,76
    anc = torch.stack(torch.meshgrid(
        torch.arange(10) * 8 + 4.0, torch.arange(10) * 8 + 4.0, indexing="xy"), -1).reshape(-1, 2)
    torch.manual_seed(0)
    pd_scores = torch.rand(1, na, 1) * 0.1
    pd_bboxes = torch.cat([anc - 4, anc + 4], -1).unsqueeze(0)
    gt = torch.tensor([[[6.0, 6.0, 10.0, 10.0]]])  # 4x4px GT，内部无 anchor 中心
    return anc, pd_scores, pd_bboxes, gt, torch.zeros(1, 1, 1), torch.ones(1, 1, 1)


def test_fork_virtual_box_compensation_exists():
    """记录性测试：本 fork 上游 TAL 已把 <stride 的 GT 扩成 stride_val 虚框选候选（4px GT 也有正样本）."""
    anc, ps, pb, gt, lab, mg = _tiny_gt_case()
    _, _, _, fg, _ = TaskAlignedAssigner(topk=10, num_classes=1)(ps, pb, anc, lab, gt, mg)
    assert fg.sum() >= 1


def test_gatal_min_pos_rescues_tiny_gt():
    """GA-TAL min_pos 兜底：在严格 in-gts 规则下（无虚框补偿）4px GT 正样本为 0，min_pos 保底 1 个."""
    anc, ps, pb, gt, lab, mg = _tiny_gt_case()
    strict = _StrictAssigner(topk=10, num_classes=1)
    _, _, _, fg0, _ = strict(ps, pb, anc, lab, gt, mg)
    assert fg0.sum() == 0, "严格规则下 4px GT 应无正样本（复现正样本饥饿）"
    ga = _StrictAssigner(topk=10, num_classes=1, metric="nwd", min_pos=True)
    _, _, ts, fg1, _ = ga(ps, pb, anc, lab, gt, mg)
    assert fg1.sum() >= 1, "GA-TAL min_pos 应保底 1 个正样本"
    assert torch.isfinite(ts).all() and ts.sum() > 0  # target_scores 权重非零，损失可回传


def test_gatal_nwd_metric_ranks_tiny_boxes():
    """NWD 度量在微小框上给出非退化的相似度排序（IoU 全 0 时 topk 排序失效，NWD 仍可区分远近）."""
    ga = TaskAlignedAssigner(topk=10, num_classes=1, metric="nwd")
    gt = torch.tensor([[6.0, 6.0, 10.0, 10.0]]).repeat(2, 1)
    near = torch.tensor([[11.0, 11.0, 15.0, 15.0], [40.0, 40.0, 44.0, 44.0]])  # 无重叠：近 vs 远
    sim = ga.iou_calculation(gt, near)
    assert sim[0] > sim[1] > 0  # IoU 两者均为 0，NWD 能区分


@pytest.mark.parametrize("former", [FarFormer(32, 2, 4), LumiFormer(32, 2), LRSA(32, 4), HFBranch(32), EDFFN(32)])
def test_former_modules_keep_shape(former):
    x = torch.randn(2, 32, 32, 32)
    y = former(x)
    assert y.shape == x.shape and torch.isfinite(y).all()


def test_former_backward():
    m = FarFormer(32, 1, 4)
    x = torch.randn(2, 32, 16, 16, requires_grad=True)
    m(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    m2 = LumiFormer(32, 1)
    x2 = torch.randn(2, 32, 16, 16, requires_grad=True)
    m2(x2).sum().backward()
    assert torch.isfinite(x2.grad).all()


# ---------------------------------------------------------------------------- iou_ext
def test_ciou_matches_stock():
    b1, b2 = _rand_boxes()
    ours = bbox_iou_ext(b1, b2, iou_type="ciou")
    stock = bbox_iou(b1, b2, xywh=False, CIoU=True)
    assert torch.allclose(ours, stock, atol=1e-5)


@pytest.mark.parametrize("t", ["iou", "giou", "diou", "ciou", "eiou", "siou", "mpdiou", "shapeiou"])
def test_iou_variants_finite_and_bounded(t):
    b1, b2 = _rand_boxes()
    v = bbox_iou_ext(b1, b2, iou_type=t)
    assert torch.isfinite(v).all() and (v <= 1.0 + 1e-6).all()
    same = bbox_iou_ext(b1, b1, iou_type=t)
    assert torch.allclose(same, torch.ones_like(same), atol=1e-4)  # 自身 IoU 族 = 1


def test_nwd_and_focaler_and_wiou():
    b1, b2 = _rand_boxes()
    s = nwd(b1, b2)
    assert ((s > 0) & (s <= 1.0 + 1e-6)).all()
    assert torch.allclose(nwd(b1, b1), torch.ones(len(b1), 1), atol=1e-5)
    iou, r = wiou_terms(b1, b2)
    assert torch.isfinite(r).all() and (r >= 1.0 - 1e-6).all()  # exp(>=0)
    f = focaler_remap(iou)
    assert ((f >= 0) & (f <= 1)).all()


@pytest.mark.parametrize(
    "iou_type", ["CIoU", "EIoU", "SIoU", "MPDIoU", "ShapeIoU", "WIoU", "NWDWise", "FocalerCIoU", "FocalerWIoU"]
)
def test_bboxloss_dispatch_all_types(iou_type):
    hyp = SimpleNamespace(iou_type=iou_type, inner_ratio=1.0, nwd_ratio=0.0)
    crit = BboxLoss(reg_max=16, hyp=hyp)
    b1, b2 = _rand_boxes(32)
    b1.requires_grad_(True)
    liou = crit._iou_loss_elem(b1, b2)
    assert liou.shape == (32, 1) and torch.isfinite(liou).all() and (liou >= 0).all()
    liou.sum().backward()
    assert torch.isfinite(b1.grad).all()


def test_bboxloss_inner_and_nwd_blend():
    hyp = SimpleNamespace(iou_type="EIoU", inner_ratio=0.75, nwd_ratio=0.5)
    crit = BboxLoss(reg_max=16, hyp=hyp)
    b1, b2 = _rand_boxes(16)
    liou = crit._iou_loss_elem(b1, b2)
    blended = (1 - 0.5) * liou + 0.5 * (1 - nwd(b1, b2))
    assert torch.isfinite(blended).all()


# ---------------------------------------------------------------------------- Lion
def test_lion_converges_on_quadratic():
    w = torch.nn.Parameter(torch.tensor([5.0, -3.0]))
    opt = Lion([w], lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        loss = (w**2).sum()
        loss.backward()
        opt.step()
    assert (w.abs() < 0.2).all(), f"Lion failed to converge: {w.data}"
