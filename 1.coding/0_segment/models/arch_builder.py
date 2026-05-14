"""模块构建器。

根据配置字典构建 backbone、neck、head。
"""

from typing import Any, Dict, List, Union

import torch.nn as nn

from .arch_registry import BACKBONE_REGISTRY, HEAD_REGISTRY, NECK_REGISTRY


class NeckSequential(nn.Module):
    """顺序执行多个 neck 的容器。

    Args:
        necks (List[nn.Module]): neck 模块列表。
    """

    def __init__(self, necks: List[nn.Module]):
        """初始化顺序 neck 容器。

        Args:
            necks (List[nn.Module]): neck 模块列表。
        """
        super().__init__()
        self.necks = nn.ModuleList(necks)

    def forward(self, features):
        """按顺序执行 neck。

        Args:
            features (Any): 输入特征，可以是列表或张量。

        Returns:
            Any: 最终输出特征。
        """
        output = features
        for neck in self.necks:
            output = neck(output)
        return output


def _build_from_cfg(
    cfg: Union[str, Dict[str, Any]], registry: Dict[str, Any], kind: str
) -> nn.Module:
    """根据配置从注册表构建模块。

    Args:
        cfg (str | Dict[str, Any]): 配置，支持名称或包含 name/args 的字典。
        registry (Dict[str, Any]): 注册表。
        kind (str): 模块类型描述，用于错误提示。

    Returns:
        nn.Module: 构建得到的模块实例。

    Raises:
        TypeError: 配置类型不支持时抛出。
        ValueError: 配置缺少 name 时抛出。
        KeyError: 注册表不存在对应模块时抛出。
    """
    if isinstance(cfg, str):
        name = cfg
        args = {}
    elif isinstance(cfg, dict):
        name = cfg.get("name")
        args = cfg.get("args") or {}
    else:
        raise TypeError(f"Unsupported {kind} config type: {type(cfg)}")

    if not name:
        raise ValueError(f"{kind} config must provide a name")

    builder = registry.get(str(name).lower())
    if builder is None:
        raise KeyError(f"{kind} not registered: {name}")

    return builder(**args)


def build_backbone(cfg: Union[str, Dict[str, Any]]) -> nn.Module:
    """构建 backbone。

    Args:
        cfg (str | Dict[str, Any]): backbone 配置。

    Returns:
        nn.Module: backbone 实例。
    """
    return _build_from_cfg(cfg, BACKBONE_REGISTRY, "backbone")


def build_neck(cfg: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> nn.Module:
    """构建 neck。

    Args:
        cfg (str | Dict[str, Any] | List[Dict[str, Any]]): neck 配置。

    Returns:
        nn.Module: neck 实例。
    """
    if isinstance(cfg, list):
        necks = [_build_from_cfg(item, NECK_REGISTRY, "neck") for item in cfg]
        return NeckSequential(necks)
    return _build_from_cfg(cfg, NECK_REGISTRY, "neck")


def build_head(cfg: Union[str, Dict[str, Any]]) -> nn.Module:
    """构建 head。

    Args:
        cfg (str | Dict[str, Any]): head 配置。

    Returns:
        nn.Module: head 实例。
    """
    return _build_from_cfg(cfg, HEAD_REGISTRY, "head")
