import warnings
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# einops 已完全移除


# ========================== 1. TSSA 模块定义 (无 einops) ==========================
class AttentionTSSA(nn.Module):
    """
    Token Statistics Self-Attention
    与原版功能一致，但完全使用原生 PyTorch 维度变换，无 einops 依赖。
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.heads = num_heads
        self.attend = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.temp = nn.Parameter(torch.ones(num_heads, 1))
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(proj_drop))

    def forward(self, x):
        # x: (B, N, dim)
        B, N, _ = x.shape
        # 原: w = rearrange(self.qkv(x), "b n (h d) -> b h n d", h=self.heads)
        qkv = self.qkv(x)  # (B, N, dim)
        w = qkv.reshape(B, N, self.heads, -1).permute(0, 2, 1, 3)  # (B, heads, N, d)
        b, h, N, d = w.shape

        w_normed = F.normalize(w, dim=-2)
        w_sq = w_normed**2
        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp)  # (B, heads, N)

        dots = torch.matmul(
            (Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w**2
        )
        attn = 1.0 / (1 + dots)
        attn = self.attn_drop(attn)
        out = -torch.mul(w.mul(Pi.unsqueeze(-1)), attn)  # (B, heads, N, d)

        # 原: out = rearrange(out, "b h n d -> b n (h d)")
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)  # (B, N, dim)
        return self.to_out(out)


# ========================== 2. 窗口重组工具 ==========================
class WindowReverser:
    """
    将 WindowPartition 后的窗口化张量还原为连续空间图
    """

    def __init__(self, ws: int, feat_h: int, feat_w: int):
        assert feat_h % ws == 0 and feat_w % ws == 0, (
            f"特征图 ({feat_h},{feat_w}) 必须能被窗口大小 ws={ws} 整除"
        )
        self.ws = ws
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.num_h = feat_h // ws
        self.num_w = feat_w // ws

    def reverse(self, x_win: torch.Tensor, is_pi: bool = False) -> torch.Tensor:
        """
        输入格式：
          - 非Pi: (B*num_win, ws*ws, C) 或 (B, num_win, ws*ws, C)
          - Pi:   (B*num_win, heads, ws*ws) 或 (B, num_win, heads, ws*ws)
        输出格式：
          - 非Pi: (B, C, H, W)
          - Pi:   (B, heads, H, W)
        """
        if not is_pi and x_win.dim() == 3:
            B = x_win.shape[0] // (self.num_h * self.num_w)
            C = x_win.shape[-1]
            x = x_win.view(B, self.num_h, self.num_w, self.ws, self.ws, C)
            x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
            return x.view(B, C, self.feat_h, self.feat_w)

        elif not is_pi and x_win.dim() == 4:
            B = x_win.shape[0]
            C = x_win.shape[-1]
            x = x_win.view(B, self.num_h, self.num_w, self.ws, self.ws, C)
            x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
            return x.view(B, C, self.feat_h, self.feat_w)

        elif is_pi and x_win.dim() == 3:
            B = x_win.shape[0] // (self.num_h * self.num_w)
            heads = x_win.shape[1]
            x = x_win.view(B, self.num_h, self.num_w, heads, self.ws, self.ws)
            x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
            return x.view(B, heads, self.feat_h, self.feat_w)

        elif is_pi and x_win.dim() == 4:
            B = x_win.shape[0]
            heads = x_win.shape[2]
            x = x_win.view(B, self.num_h, self.num_w, heads, self.ws, self.ws)
            x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
            return x.view(B, heads, self.feat_h, self.feat_w)

        else:
            raise ValueError(f"不支持的窗口化张量 shape={x_win.shape}, is_pi={is_pi}")


# ========================== 3. 可视化器 (无 einops) ==========================
class S2BlockVisualizer:
    """
    针对 S²-Block (WindowPartition -> VSSD -> TSSA -> WindowReverse) 的诊断可视化
    """

    def __init__(
        self,
        layer_name: str,
        ws: int,
        feat_hw: Tuple[int, int],
        num_heads_show: int = 4,
        gamma: float = 0.4,
    ):
        self.layer_name = layer_name
        self.ws = ws
        self.feat_h, self.feat_w = feat_hw
        self.num_heads_show = num_heads_show
        self.gamma = gamma
        self.reverser = WindowReverser(ws, self.feat_h, self.feat_w)

    def extract_pi(
        self, tssa_module: nn.Module, vssd_output: torch.Tensor
    ) -> torch.Tensor:
        """
        从 VSSD 输出（即 TSSA 输入）重算 Pi，并重组为空间图 (B, heads, H, W)
        使用原生 reshape/permute 替代 einops。
        """
        with torch.no_grad():
            inp = vssd_output  # (B*num_win, ws*ws, dim)
            B_tot, N, dim = inp.shape
            heads = tssa_module.heads
            # 原: w = rearrange(tssa_module.qkv(vssd_output), "b n (h d) -> b h n d", h=heads)
            qkv = tssa_module.qkv(inp)  # (B_tot, N, dim)
            w = qkv.reshape(B_tot, N, heads, -1).permute(
                0, 2, 1, 3
            )  # (B_tot, heads, N, d)

            w_normed = F.normalize(w, dim=-2)
            w_sq = w_normed**2
            pi_logits = torch.sum(w_sq, dim=-1) * tssa_module.temp  # (B_tot, heads, N)
            pi = tssa_module.attend(pi_logits)
            return self.reverser.reverse(pi, is_pi=True)

    def plot(
        self,
        pi_spatial: torch.Tensor,
        vssd_output: Optional[torch.Tensor] = None,
        original_img: Optional[np.ndarray] = None,
        gt_bboxes: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ):
        """
        绘制诊断面板
        """
        B, heads, fh, fw = pi_spatial.shape
        assert B == 1, "仅支持 batch_size=1 可视化"
        n_show = min(self.num_heads_show, heads)

        ncols = n_show + 1
        if original_img is not None:
            ncols += 1
        if vssd_output is not None:
            ncols += 1

        fig, axes = plt.subplots(1, ncols, figsize=(3.0 * ncols, 3.2))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = np.atleast_1d(axes)
        ax_idx = 0

        # ① 原图 + GT 框
        if original_img is not None:
            ax = axes[ax_idx]
            ax.imshow(original_img)
            if gt_bboxes is not None:
                for box in gt_bboxes:
                    ax.add_patch(
                        plt.Rectangle(
                            (box[0], box[1]),
                            box[2] - box[0],
                            box[3] - box[1],
                            fill=False,
                            edgecolor="red",
                            linewidth=2,
                        )
                    )
            ax.set_title("Input + GT")
            ax.axis("off")
            ax_idx += 1

        # ② VSSD 响应（带窗口网格线，诊断边界伪影）
        if vssd_output is not None:
            vssd_spatial = self.reverser.reverse(vssd_output, is_pi=False)
            vssd_map = vssd_spatial[0].abs().mean(dim=0).cpu().numpy()

            ax = axes[ax_idx]
            im = ax.imshow(vssd_map, cmap="magma")
            for i in range(1, self.reverser.num_h):
                ax.axhline(i * self.ws - 0.5, color="cyan", lw=0.8, alpha=0.6)
            for j in range(1, self.reverser.num_w):
                ax.axvline(j * self.ws - 0.5, color="cyan", lw=0.8, alpha=0.6)
            ax.set_title("VSSD Response")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax_idx += 1

        # ③-⑥ TSSA Heads 软聚类
        for k in range(n_show):
            pi_k = pi_spatial[0, k].cpu().numpy()
            # 小目标增强：gamma 变换放大低响应区
            vmax = np.percentile(pi_k, 99.5)
            pi_k = np.clip(pi_k, 0, vmax) ** self.gamma

            ax = axes[ax_idx + k]
            im = ax.imshow(pi_k, cmap="viridis", interpolation="nearest")

            # GT 框映射到特征图尺度
            if original_img is not None and gt_bboxes is not None:
                scale_x = fw / original_img.shape[1]
                scale_y = fh / original_img.shape[0]
                for box in gt_bboxes:
                    rect = plt.Rectangle(
                        (box[0] * scale_x, box[1] * scale_y),
                        (box[2] - box[0]) * scale_x,
                        (box[3] - box[1]) * scale_y,
                        fill=False,
                        edgecolor="red",
                        linewidth=1.5,
                    )
                    ax.add_patch(rect)

            ax.set_title(f"Head {k}")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax_idx += n_show

        plt.suptitle(
            f"{self.layer_name} | ws={self.ws} | feat={self.feat_h}×{self.feat_w}",
            fontsize=12,
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


# ========================== 4. Hook 提取器 ==========================
class TSSAHookExtractor:
    """
    通过 PyTorch Hook 自动遍历模型，捕获所有 AttentionTSSA 模块
    """

    def __init__(self, visualizers: List[S2BlockVisualizer]):
        self.visualizers = visualizers
        self.captured: List[Dict] = []
        self._handles = []
        self._idx = 0

    def register(self, model: nn.Module, target_class: str = "AttentionTSSA"):
        found = 0
        for name, module in model.named_modules():
            if module.__class__.__name__ == target_class:
                if self._idx >= len(self.visualizers):
                    warnings.warn(f"TSSA 层 '{name}' 无对应 visualizer，跳过")
                    continue
                viz = self.visualizers[self._idx]
                h = module.register_forward_hook(self._make_hook(module, viz, name))
                self._handles.append(h)
                print(
                    f"[Hook] #{self._idx} {name}  "
                    f"ws={viz.ws} feat={viz.feat_h}×{viz.feat_w}"
                )
                self._idx += 1
                found += 1

        if found == 0:
            raise RuntimeError(f"模型中未找到 {target_class}")
        if found != len(self.visualizers):
            warnings.warn(f"找到 {found} 个 TSSA，但配置了 {len(self.visualizers)} 个")

    def _make_hook(self, module, viz, name):
        def hook(m, inp, out):
            self.captured.append(
                {
                    "name": name,
                    "viz": viz,
                    "module": m,
                    "vssd_out": inp[0].detach(),
                    "tssa_out": out.detach(),
                }
            )

        return hook

    def visualize_all(
        self,
        original_img: Optional[np.ndarray] = None,
        gt_bboxes: Optional[np.ndarray] = None,
        save_prefix: str = "tssa_vis",
    ):
        for i, cap in enumerate(self.captured):
            viz = cap["viz"]
            pi = viz.extract_pi(cap["module"], cap["vssd_out"])
            path = f"{save_prefix}_layer{i}_{cap['name'].replace('.', '_')}.png"
            viz.plot(
                pi,
                vssd_output=cap["vssd_out"],
                original_img=original_img,
                gt_bboxes=gt_bboxes,
                save_path=path,
            )
            print(f"[Save] {path}")

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.captured.clear()
        self._idx = 0


# ========================== 5. 详细用法示例 ==========================
def demo_usage():
    # 构建一个轻量模拟模型（使用修改后的 AttentionTSSA）
    class MockS2Block(nn.Module):
        def __init__(self, dim, ws, feat_hw):
            super().__init__()
            self.ws, self.fh, self.fw = ws, feat_hw[0], feat_hw[1]
            self.norm_vssd = nn.LayerNorm(dim)
            self.vssd = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
            self.norm_tssa = nn.LayerNorm(dim)
            self.tssa = AttentionTSSA(dim, num_heads=8)
            self.norm_mlp = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
            )

        def forward(self, x):
            B, C, H, W = x.shape
            # WindowPartition
            x = x.view(B, C, self.fh // self.ws, self.ws, self.fw // self.ws, self.ws)
            x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
            x = x.view(
                B * self.fh * self.fw // (self.ws * self.ws), self.ws * self.ws, C
            )
            # S²-Block 内部
            x = x + self.vssd(self.norm_vssd(x))
            x = x + self.tssa(self.norm_tssa(x))
            x = x + self.mlp(self.norm_mlp(x))
            # WindowReverse
            x = x.view(B, self.fh // self.ws, self.fw // self.ws, self.ws, self.ws, C)
            x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
            return x.view(B, C, self.fh, self.fw)

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.s2_p4 = MockS2Block(256, ws=8, feat_hw=(40, 40))
            self.s2_p5 = MockS2Block(512, ws=4, feat_hw=(20, 20))

        def forward(self, x):
            feat_p4 = torch.randn(1, 256, 40, 40).cuda()
            out_p4 = self.s2_p4(feat_p4)
            feat_p5 = torch.randn(1, 512, 20, 20).cuda()
            out_p5 = self.s2_p5(feat_p5)
            return out_p5

    model = MockModel().eval().cuda()

    configs = [
        {
            "layer_name": "backbone.s2_p4",
            "ws": 8,
            "feat_hw": (40, 40),
            "num_heads_show": 4,
            "gamma": 0.4,
        },
        {
            "layer_name": "backbone.s2_p5",
            "ws": 4,
            "feat_hw": (20, 20),
            "num_heads_show": 4,
            "gamma": 0.3,
        },
    ]
    visualizers = [S2BlockVisualizer(**cfg) for cfg in configs]

    extractor = TSSAHookExtractor(visualizers)
    extractor.register(model, target_class="AttentionTSSA")

    dummy_input = torch.randn(1, 3, 640, 640).cuda()
    with torch.no_grad():
        _ = model(dummy_input)

    extractor.visualize_all(
        original_img=None,
        gt_bboxes=None,
        save_prefix="my_model",
    )

    extractor.remove()
    print("完成。所有 Hook 已注销，显存已释放。")


if __name__ == "__main__":
    demo_usage()
