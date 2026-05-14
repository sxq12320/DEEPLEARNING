"""TS-Dual 分割 + 检测联合模型。"""

from copy import deepcopy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .arch_builder import build_backbone, build_head, build_neck


class TSDualSegDetNet(nn.Module):
    """TS-Dual 网络组合体。

    结构：backbone → neck → head
    输出包含 bbox 预测与 mask logits。

    Args:
        model_cfg (dict): 模型配置，包含 backbone/neck/head 三部分。
    """

    def __init__(self, model_cfg: dict):
        """初始化 TS-Dual 网络。

        Args:
            model_cfg (dict): 模型配置字典。
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.backbone = build_backbone(model_cfg["backbone"])
        self.neck = build_neck(model_cfg["neck"])
        self.head = build_head(model_cfg["head"])

    def forward(
        self,
        rgb: Union[torch.Tensor, dict],
        prompt: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
    ) -> dict:
        """前向传播。

        Args:
            rgb (torch.Tensor | dict): RGB 输入或输入字典。
            prompt (torch.Tensor | None): Mask 先验提示。
            depth (torch.Tensor | None): Depth 输入。

        Returns:
            dict: {"bbox": bbox_pred, "mask": mask_logits}。
        """
        rgb, prompt, depth = self._parse_inputs(rgb, prompt, depth)
        features = self.backbone(rgb, prompt, depth)
        neck_out = self.neck(features)
        bbox_pred, mask_logits = self.head(neck_out, input_shape=rgb.shape[2:])
        return {"bbox": bbox_pred, "mask": mask_logits}

    def _parse_inputs(
        self,
        rgb: Union[torch.Tensor, dict],
        prompt: Optional[torch.Tensor],
        depth: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """解析输入，兼容 dict 形式。

        Args:
            rgb (torch.Tensor | dict): RGB 输入或包含多模态的字典。
            prompt (torch.Tensor | None): Mask 先验提示。
            depth (torch.Tensor | None): Depth 输入。

        Returns:
            tuple: (rgb, prompt, depth)。
        """
        if isinstance(rgb, dict):
            depth = rgb.get("depth")
            prompt = rgb.get("prompt")
            rgb = rgb.get("rgb")
        if rgb is None:
            raise ValueError("RGB 输入不能为空")
        return rgb, prompt, depth


def build_ts_dual_model(
    model_cfg: dict, num_classes: Optional[int] = None
) -> TSDualSegDetNet:
    """根据配置构建 TS-Dual 模型。

    Args:
        model_cfg (dict): 模型配置字典。
        num_classes (int | None): 分割类别数（用于覆盖 head 输出通道）。

    Returns:
        TSDualSegDetNet: 构建后的模型。
    """
    cfg = deepcopy(model_cfg)
    if num_classes is not None:
        head_args = cfg.setdefault("head", {}).setdefault("args", {})
        head_args["mask_out_ch"] = num_classes
    return TSDualSegDetNet(cfg)
