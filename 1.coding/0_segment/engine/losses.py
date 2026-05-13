"""损失函数模块。

提供分割任务（SegmentationLoss）与目标检测任务（YOLODetectionLoss）
的损失计算功能。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# 分割损失
# ======================================================================
class SegmentationLoss(nn.Module):
    """分割任务损失函数包装器，默认使用 BCEWithLogitsLoss。

    该类根据损失类型初始化不同的损失进行计算。

    Attributes:
        loss_type (str): 损失类型（"bce" 或 "cross_entropy"）。
        **kwargs: 传给具体损失函数的参数。
    """

    def __init__(self, loss_type="bce", **kwargs):
        """初始化损失函数模块。"""
        super().__init__()
        if loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss(**kwargs)
        elif loss_type == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss(**kwargs)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, pred, target):
        """计算损失。

        Args:
            pred (torch.Tensor): 预测 logits。
            target (torch.Tensor): 目标掩码。

        Returns:
            torch.Tensor: 损失值。
        """
        return self.criterion(pred, target)


# ======================================================================
# YOLO 检测损失辅助函数
# ======================================================================
def make_anchors(feats, strides, grid_cell_offset=0.5):
    """根据多尺度特征图尺寸生成锚点坐标与步幅张量。

    对每个检测层，在其特征图的每个网格中心处生成一个锚点，
    用于后续的预测框解码与标签分配。

    Args:
        feats (List[torch.Tensor]): 多尺度特征图列表。
        strides (List[int]): 各层相对于原图的下采样倍率。
        grid_cell_offset (float): 网格单元中心偏移。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            anchor_points — (N_anchors, 2) 的 (x, y) 锚点坐标；
            stride_tensor  — (N_anchors, 1) 的对应步幅。
    """
    anchor_points = []
    stride_tensor = []
    for feat, stride in zip(feats, strides):
        h, w = feat.shape[2], feat.shape[3]
        # 网格中心坐标
        sx = torch.arange(w, dtype=feat.dtype, device=feat.device) + grid_cell_offset
        sy = torch.arange(h, dtype=feat.dtype, device=feat.device) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack([sx, sy], dim=-1).reshape(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=feat.dtype, device=feat.device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(pred_dist, anchor_points, stride):
    """将 DFL 分布预测解码为边界框坐标。

    分布通过 softmax 加权求和转化为 ltrb 偏移量，
    再结合锚点中心坐标和步幅还原为绝对 xyxy 坐标。

    Args:
        pred_dist (torch.Tensor): 预测分布 (B, N_anchors, 4*reg_max)。
        anchor_points (torch.Tensor): 锚点中心 (N_anchors, 2)。
        stride (torch.Tensor): 步幅 (N_anchors, 1) 或标量。

    Returns:
        torch.Tensor: 解码后的边界框 (B, N_anchors, 4)，xyxy 格式。
    """
    reg_max = pred_dist.shape[-1] // 4
    pred_dist = pred_dist.reshape(*pred_dist.shape[:-1], 4, reg_max)
    pred_dist = pred_dist.softmax(dim=-1)
    # ltrb 偏移量：用区间索引加权求和
    integral = torch.arange(reg_max, dtype=pred_dist.dtype, device=pred_dist.device)
    stride = stride.view(1, -1, 1)  # (1, N, 1) 供广播
    ltrb = (pred_dist * integral).sum(dim=-1) * stride
    # 转换为 xyxy
    anchor = anchor_points.unsqueeze(0)  # (1, N, 2)
    x1y1 = anchor - ltrb[..., :2]
    x2y2 = anchor + ltrb[..., 2:]
    return torch.cat([x1y1, x2y2], dim=-1)


def bbox2dist(anchor_points, bbox, reg_max):
    """将边界框转换为 DFL 分布目标。

    把 xyxy 格式的框转换为相对于锚点中心的 ltrb 偏移量，
    再量化为 [0, reg_max-1] 区间内的整数值。

    Args:
        anchor_points (torch.Tensor): 锚点中心 (N, 2)。
        bbox (torch.Tensor): 真实框 (N, 4)，xyxy 格式。
        reg_max (int): 分布的区间数量。

    Returns:
        torch.Tensor: DFL 目标索引 (N, 4)，取值 [0, reg_max-1]。
    """
    ltrb_offset = torch.cat([
        anchor_points - bbox[..., :2],
        bbox[..., 2:] - anchor_points,
    ], dim=-1)
    return ltrb_offset.clamp(0, reg_max - 1).long()


# ======================================================================
# TaskAlignedAssigner
# ======================================================================
class TaskAlignedAssigner(nn.Module):
    """任务对齐标签分配器（Task-Aligned Assigner）。

    基于预测的分类得分与预测框-真实框 IoU 的加权乘积
    （alignment_metric = cls_score^alpha * IoU^beta）选择正样本，
    为每个真实框选出 topk 个最高对齐度的锚点作为正样本。

    Attributes:
        topk (int): 每个真实框保留的正样本数量。
        alpha (float): 分类得分的指数权重。
        beta (float): IoU 项的指数权重。
    """

    def __init__(self, topk=13, alpha=1.0, beta=6.0):
        """初始化分配器。"""
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-9

    @torch.no_grad()
    def forward(self, pred_scores, pred_bboxes, gt_labels, gt_bboxes,
                anchor_points, stride):
        """执行标签分配。

        Args:
            pred_scores (torch.Tensor): 预测分类得分 (B, N, num_classes)。
            pred_bboxes (torch.Tensor): 预测框 (B, N, 4)，xyxy 格式。
            gt_labels (torch.Tensor): 真实类别 (B, M, 1) 或 (B, M)。
            gt_bboxes (torch.Tensor): 真实框 (B, M, 4)。
            anchor_points (torch.Tensor): 锚点坐标 (N, 2)。
            stride (torch.Tensor): 步幅 (N, 1)。

        Returns:
            Tuple:
                target_labels   — (B, N) 目标类别，背景为 num_classes；
                target_bboxes   — (B, N, 4) 目标框；
                target_scores   — (B, N) 目标得分（对齐度）；
                fg_mask         — (B, N) 前景掩码。
        """
        bs, num_anchors = pred_scores.shape[:2]
        num_gts = gt_bboxes.shape[1]

        device = pred_scores.device
        target_labels = torch.full((bs, num_anchors), pred_scores.shape[-1],
                                   dtype=torch.int64, device=device)
        target_bboxes = torch.zeros(bs, num_anchors, 4, device=device)
        target_scores = torch.zeros(bs, num_anchors, device=device)
        fg_mask = torch.zeros(bs, num_anchors, dtype=torch.bool, device=device)

        for b in range(bs):
            valid_gt = gt_bboxes[b].sum(dim=-1) > 0
            num_valid = valid_gt.sum().item()
            if num_valid == 0:
                continue

            gt_box = gt_bboxes[b][valid_gt]
            gt_label = gt_labels[b][valid_gt].squeeze(-1) if gt_labels[b].ndim == 2 else gt_labels[b][valid_gt]

            # ---- 计算 pairwise IoU ----
            lt = torch.max(pred_bboxes[b, :, None, :2], gt_box[None, :, :2])
            rb = torch.min(pred_bboxes[b, :, None, 2:], gt_box[None, :, 2:])
            wh = (rb - lt).clamp(min=0)
            inter = wh[..., 0] * wh[..., 1]
            area_p = (pred_bboxes[b, :, 2] - pred_bboxes[b, :, 0]) * (pred_bboxes[b, :, 3] - pred_bboxes[b, :, 1])
            area_g = (gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1])
            iou = inter / (area_p[:, None] + area_g[None, :] - inter + self.eps)

            # ---- 分类得分 ----
            cls_score = pred_scores[b].sigmoid().max(dim=-1)[0]  # (N,)

            # ---- 对齐度 ----
            align_metric = (cls_score[:, None] ** self.alpha) * (iou ** self.beta)
            # 过滤无效锚点（框中心超出 GT 范围的锚点）
            gt_centers = (gt_box[:, :2] + gt_box[:, 2:]) / 2.0
            anchor_in_gt = (anchor_points[:, 0] >= gt_centers[:, 0].min()) & \
                           (anchor_points[:, 0] <= gt_centers[:, 0].max()) & \
                           (anchor_points[:, 1] >= gt_centers[:, 1].min()) & \
                           (anchor_points[:, 1] <= gt_centers[:, 1].max())
            align_metric = align_metric * anchor_in_gt[:, None].float()

            # ---- topk 选择 ----
            topk_mask = torch.zeros_like(align_metric, dtype=torch.bool)
            topk_metrics, topk_idxs = torch.topk(
                align_metric, min(self.topk, num_anchors), dim=0, sorted=False
            )
            topk_mask.scatter_(0, topk_idxs, 1)

            # ---- 每个锚点只分配给对齐度最高的 GT ----
            align_metric[~topk_mask] = 0
            matched_gt = align_metric.argmax(dim=1)  # (N,)
            matched_scores = align_metric[torch.arange(num_anchors), matched_gt]
            matched_valid = matched_scores > 0

            target_labels[b][matched_valid] = gt_label[matched_gt[matched_valid]].long()
            target_bboxes[b][matched_valid] = gt_box[matched_gt[matched_valid]]
            target_scores[b][matched_valid] = matched_scores[matched_valid]
            fg_mask[b] = matched_valid

        return target_labels, target_bboxes, target_scores, fg_mask


# ======================================================================
# CIoU Loss
# ======================================================================
def ciou_loss(pred_bboxes, target_bboxes, eps=1e-7):
    """计算 CIoU（Complete IoU）损失。

    CIoU 在 IoU 的基础上考虑了中心点距离和宽高比一致性，
    相比 GIoU / DIoU 能更好指导框回归的收敛方向和速度。

    Args:
        pred_bboxes (torch.Tensor): 预测框 (N, 4)，xyxy 格式。
        target_bboxes (torch.Tensor): 真实框 (N, 4)，xyxy 格式。
        eps (float): 数值稳定项。

    Returns:
        torch.Tensor: 标量 CIoU 损失。
    """
    # IoU
    lt = torch.max(pred_bboxes[..., :2], target_bboxes[..., :2])
    rb = torch.min(pred_bboxes[..., 2:], target_bboxes[..., 2:])
    wh = (rb - lt).clamp(min=eps)
    inter = wh[..., 0] * wh[..., 1]
    area_p = (pred_bboxes[..., 2] - pred_bboxes[..., 0]) * (pred_bboxes[..., 3] - pred_bboxes[..., 1])
    area_g = (target_bboxes[..., 2] - target_bboxes[..., 0]) * (target_bboxes[..., 3] - target_bboxes[..., 1])
    union = area_p + area_g - inter + eps
    iou = inter / union

    # 中心距离 / 闭包对角线
    c_p = (pred_bboxes[..., :2] + pred_bboxes[..., 2:]) / 2.0
    c_g = (target_bboxes[..., :2] + target_bboxes[..., 2:]) / 2.0
    rho2 = ((c_p - c_g) ** 2).sum(dim=-1)
    enclose_lt = torch.min(pred_bboxes[..., :2], target_bboxes[..., :2])
    enclose_rb = torch.max(pred_bboxes[..., 2:], target_bboxes[..., 2:])
    c2 = ((enclose_rb - enclose_lt) ** 2).sum(dim=-1) + eps

    # 宽高比一致性 v
    w_p, h_p = pred_bboxes[..., 2] - pred_bboxes[..., 0], pred_bboxes[..., 3] - pred_bboxes[..., 1]
    w_g, h_g = target_bboxes[..., 2] - target_bboxes[..., 0], target_bboxes[..., 3] - target_bboxes[..., 1]
    v = (4.0 / (math.pi ** 2)) * ((torch.atan(w_g / (h_g + eps)) - torch.atan(w_p / (h_p + eps))) ** 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return 1.0 - iou + rho2 / c2 + alpha * v


# ======================================================================
# DFL Loss
# ======================================================================
def distribution_focal_loss(pred_dist, target_idx, reg_max, weight=None, eps=1e-7):
    """Distribution Focal Loss。

    将边界框回归建模为离散分布，损失使预测分布向目标索引相邻区间集中，
    缓解框坐标的模糊性和不确定性。

    Args:
        pred_dist (torch.Tensor): 预测分布 (N, 4*reg_max)。
        target_idx (torch.Tensor): 目标离散索引 (N, 4)，long tensor。
        reg_max (int): 分布的区间数量。
        weight (torch.Tensor | None): 逐样本权重 (N,)。
        eps (float): 数值稳定项。

    Returns:
        torch.Tensor: 标量 DFL 损失。
    """
    pred_dist = pred_dist.reshape(-1, reg_max)
    target_idx = target_idx.clamp(0, reg_max - 1).reshape(-1)

    # 目标左右相邻区间
    left = target_idx
    right = (target_idx + 1).clamp(max=reg_max - 1)

    loss_left = F.cross_entropy(pred_dist, left, reduction='none')
    loss_right = F.cross_entropy(pred_dist, right, reduction='none')
    # 线性插值权重：目标越靠近整数，权重越集中
    w_right = (target_idx.float() - left.float())
    w_left = (right.float() - target_idx.float())

    dfl = loss_left * w_left + loss_right * w_right
    if weight is not None:
        dfl = dfl.reshape(*weight.shape, 4) * weight.unsqueeze(-1)
    return dfl.mean()


# ======================================================================
# YOLO 检测总损失
# ======================================================================
class YOLODetectionLoss(nn.Module):
    """YOLO11 检测任务综合损失函数。

    整合三类子损失：
        box_loss  — CIoU 框回归损失；
        cls_loss  — BCE 分类损失；
        dfl_loss  — Distribution Focal Loss。

    使用 TaskAlignedAssigner 动态分配正负样本。

    Attributes:
        num_classes (int): 类别数。
        reg_max (int): DFL 区间数。
        strides (List[int]): 各检测层的下采样倍率。
        tal_topk (int): 分配器 topk 参数。
        box_gain (float): 框损失权重。
        cls_gain (float): 分类损失权重。
        dfl_gain (float): DFL 损失权重。
    """

    def __init__(self, num_classes=80, reg_max=16, strides=(8, 16, 32),
                 tal_topk=13, box_gain=7.5, cls_gain=0.5, dfl_gain=1.5):
        """初始化检测损失函数。"""
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.assigner = TaskAlignedAssigner(topk=tal_topk)

    def forward(self, cls_preds, reg_preds, targets, features):
        """计算检测综合损失。

        Args:
            cls_preds (List[torch.Tensor]): 每个检测层的分类 logits，
                                            形状 (B, num_classes, Hi, Wi)。
            reg_preds (List[torch.Tensor]): 每个检测层的回归输出，
                                            形状 (B, 4*reg_max, Hi, Wi)。
            targets (List[dict]): 批次的真实标注，每个元素为 dict:
                { 'labels': (M_i,) long tensor,
                  'boxes':  (M_i, 4) tensor }，boxes 为 xyxy 归一化坐标 [0,1]。
            features (List[torch.Tensor]): backbone 特征（用于计算锚点尺寸）。

        Returns:
            Tuple[torch.Tensor, dict]:
                total_loss — 总损失标量；
                loss_items — 各项损失字典 {'box': ..., 'cls': ..., 'dfl': ...}。
        """
        device = cls_preds[0].device
        bs = cls_preds[0].shape[0]

        # ---- 生成锚点 ----
        anchor_points, stride_tensor = make_anchors(features, self.strides)

        # ---- 将各层预测拼接为扁平张量 ----
        all_cls = []
        all_reg = []
        for i, (cp, rp) in enumerate(zip(cls_preds, reg_preds)):
            b, c, h, w = cp.shape
            all_cls.append(cp.permute(0, 2, 3, 1).reshape(b, h * w, c))
            all_reg.append(rp.permute(0, 2, 3, 1).reshape(b, h * w, 4 * self.reg_max))
        pred_cls = torch.cat(all_cls, dim=1)  # (B, N, num_classes)
        pred_reg = torch.cat(all_reg, dim=1)  # (B, N, 4*reg_max)

        # ---- 批量解码预测框 ----
        pred_boxes = dist2bbox(pred_reg, anchor_points, stride_tensor)  # (B, N, 4)

        # ---- 构建真实标注张量 ----
        max_gts = max(len(t['labels']) for t in targets) if targets else 0
        if max_gts == 0:
            # 无可用的真实框，返回零损失
            return torch.tensor(0.0, device=device, requires_grad=True), {
                'box': 0.0, 'cls': 0.0, 'dfl': 0.0
            }

        gt_labels = torch.zeros(bs, max_gts, dtype=torch.int64, device=device)
        gt_bboxes = torch.zeros(bs, max_gts, 4, device=device)
        for i, t in enumerate(targets):
            m = len(t['labels'])
            if m > 0:
                gt_labels[i, :m] = t['labels'].to(device)
                # 将归一化坐标转为像素坐标
                img_h, img_w = features[0].shape[2] * self.strides[0], features[0].shape[3] * self.strides[0]
                boxes = t['boxes'].to(device).clone()
                boxes[:, [0, 2]] *= img_w
                boxes[:, [1, 3]] *= img_h
                gt_bboxes[i, :m] = boxes

        # ---- 标签分配 ----
        pred_scores = pred_cls.detach().sigmoid()
        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores, pred_boxes.detach(), gt_labels, gt_bboxes,
            anchor_points, stride_tensor
        )

        if fg_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                'box': 0.0, 'cls': 0.0, 'dfl': 0.0
            }

        # ---- 分类损失 (BCE) ----
        cls_target = F.one_hot(target_labels.clamp_max(self.num_classes),
                               self.num_classes + 1)[..., :-1].float()  # drop background class
        cls_target = cls_target * target_scores.unsqueeze(-1)
        cls_loss = self.bce(pred_cls, cls_target).sum(dim=-1)  # (B, N)
        cls_loss = (cls_loss * fg_mask.float()).sum() / max(fg_mask.sum(), 1)

        # ---- 框回归损失 (CIoU) ----
        fg_idx = fg_mask.nonzero(as_tuple=True)
        if fg_idx[0].numel() > 0:
            pred_boxes_fg = pred_boxes[fg_idx]
            target_boxes_fg = target_bboxes[fg_idx]
            ciou = ciou_loss(pred_boxes_fg, target_boxes_fg)
            box_loss = (ciou * target_scores[fg_idx]).sum() / max(target_scores[fg_idx].sum(), 1)
        else:
            box_loss = torch.tensor(0.0, device=device)

        # ---- DFL 损失 ----
        if fg_idx[0].numel() > 0:
            pred_dist_fg = pred_reg[fg_idx]
            anchor_fg = anchor_points[fg_idx[1]]
            stride_fg = stride_tensor[fg_idx[1]]
            # 将目标框转为相对锚点的距离并量化
            ltrb_target = bbox2dist(anchor_fg, target_boxes_fg / stride_fg, self.reg_max)
            dfl_loss = distribution_focal_loss(
                pred_dist_fg, ltrb_target, self.reg_max,
                weight=target_scores[fg_idx]
            )
        else:
            dfl_loss = torch.tensor(0.0, device=device)

        # ---- 总损失 ----
        box_loss = box_loss * self.box_gain
        cls_loss = cls_loss * self.cls_gain
        dfl_loss = dfl_loss * self.dfl_gain
        total_loss = box_loss + cls_loss + dfl_loss

        return total_loss, {
            'box': box_loss.detach().item() if torch.is_tensor(box_loss) else box_loss,
            'cls': cls_loss.detach().item() if torch.is_tensor(cls_loss) else cls_loss,
            'dfl': dfl_loss.detach().item() if torch.is_tensor(dfl_loss) else dfl_loss,
        }
