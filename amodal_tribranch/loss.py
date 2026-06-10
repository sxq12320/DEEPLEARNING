"""
Physics-aware Weakly-supervised Tri-Branch Amodal Loss.

Composite loss function with 6 components:
  1. L_sup_full:   Strong supervision — BCE + Dice on pred_full vs Amodal_GT
  2. L_weak:       Weak supervision — L1 on pred_vis/pred_occ vs pseudo labels (low weight)
  3. L_excl:       Exclusivity — P_vis * P_occ should be ~0 (no overlap)
  4. L_subset:     Subset constraint — vis/occ should not exceed Amodal_GT
  5. L_union:      Union consistency — P_full ≈ P_vis + P_occ - P_vis*P_occ
  6. L_edge:       Cross-modal edge alignment — boundary(P_vis) ≈ rgb_edges
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TriBranchAmodalLoss(nn.Module):
    """Physics-aware weakly-supervised loss for tri-branch amodal segmentation."""

    def __init__(
        self,
        w_sup=1.0,
        w_weak=0.2,
        w_excl=0.5,
        w_subset=0.5,
        w_union=0.3,
        w_edge=0.2,
        dice_smooth=1e-5,
    ):
        """
        Args:
            w_sup:     weight for strong supervision loss (Full vs Amodal_GT)
            w_weak:    weight for weak supervision loss (Vis/Occ vs pseudo labels)
            w_excl:    weight for exclusivity constraint (vis ∩ occ ≈ 0)
            w_subset:  weight for subset constraint (vis ⊆ GT, occ ⊆ GT)
            w_union:   weight for union consistency (full ≈ vis ∪ occ)
            w_edge:    weight for cross-modal edge alignment
            dice_smooth: smoothing factor for Dice loss (numerical stability)
        """
        super().__init__()
        self.w_sup = w_sup
        self.w_weak = w_weak
        self.w_excl = w_excl
        self.w_subset = w_subset
        self.w_union = w_union
        self.w_edge = w_edge
        self.dice_smooth = dice_smooth

        # Pre-define Sobel kernels for boundary extraction (registered as buffers)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _bce_loss(self, pred, target):
        """Binary cross-entropy loss with logits."""
        return F.binary_cross_entropy_with_logits(pred, target, reduction='mean')

    def _dice_loss(self, pred, target):
        """Dice loss (computed on sigmoid probabilities).

        Args:
            pred: logits [B, 1, H, W]
            target: binary mask [B, 1, H, W]
        """
        pred_prob = torch.sigmoid(pred)
        intersection = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.dice_smooth) / (union + self.dice_smooth)
        return 1.0 - dice.mean()

    def _extract_boundary(self, x):
        """Extract spatial boundary from a probability map using Sobel filter.

        Args:
            x: probability map [B, 1, H, W], values in [0, 1]

        Returns:
            boundary: [B, 1, H, W], gradient magnitude
        """
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        boundary = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return boundary

    def forward(self, pred_vis, pred_occ, pred_full, amodal_gt, pseudo_vis, pseudo_occ, rgb_edges):
        """Compute the composite tri-branch amodal loss.

        Args:
            pred_vis:   [B, 1, H, W] visible mask logits (from model)
            pred_occ:   [B, 1, H, W] occluded mask logits (from model)
            pred_full:  [B, 1, H, W] full amodal mask logits (from model)
            amodal_gt:  [B, 1, H, W] ground truth amodal mask (binary, 0/1)
            pseudo_vis: [B, 1, H, W] pseudo visible mask (from depth, binary)
            pseudo_occ: [B, 1, H, W] pseudo occluded mask (from depth, binary)
            rgb_edges:  [B, 1, H, W] RGB edge map (from Canny, 0~1)

        Returns:
            total_loss: scalar, weighted sum of all loss components
            loss_dict:  dict of individual loss values for logging
        """
        # Convert logits to probabilities for constraint losses
        P_vis = torch.sigmoid(pred_vis)
        P_occ = torch.sigmoid(pred_occ)
        P_full = torch.sigmoid(pred_full)

        # ============================================================
        # 1. Strong supervision: pred_full vs Amodal_GT (BCE + Dice)
        # ============================================================
        L_bce = self._bce_loss(pred_full, amodal_gt)
        L_dice = self._dice_loss(pred_full, amodal_gt)
        L_sup_full = L_bce + L_dice

        # ============================================================
        # 2. Weak supervision: pred_vis/pred_occ vs pseudo labels (L1)
        # ============================================================
        L_weak_vis = F.l1_loss(P_vis, pseudo_vis)
        L_weak_occ = F.l1_loss(P_occ, pseudo_occ)
        L_weak = L_weak_vis + L_weak_occ

        # ============================================================
        # 3. Exclusivity constraint: P_vis * P_occ → 0
        # ============================================================
        L_excl = (P_vis * P_occ).mean()

        # ============================================================
        # 4. Subset constraint: vis ⊆ GT, occ ⊆ GT
        #    ReLU(P_vis - GT) + ReLU(P_occ - GT) → 0
        # ============================================================
        L_subset = F.relu(P_vis - amodal_gt).mean() + F.relu(P_occ - amodal_gt).mean()

        # ============================================================
        # 5. Union consistency: P_full ≈ P_vis + P_occ - P_vis*P_occ
        #    Use P_full.detach() so this loss only regularizes vis/occ
        # ============================================================
        P_union = P_vis + P_occ - P_vis * P_occ
        L_union = F.l1_loss(P_full.detach(), P_union)

        # ============================================================
        # 6. Cross-modal edge alignment: boundary(P_vis) ≈ rgb_edges
        # ============================================================
        vis_boundary = self._extract_boundary(P_vis)
        L_edge = F.l1_loss(vis_boundary, rgb_edges)

        # ============================================================
        # Total loss
        # ============================================================
        total_loss = (
            self.w_sup * L_sup_full
            + self.w_weak * L_weak
            + self.w_excl * L_excl
            + self.w_subset * L_subset
            + self.w_union * L_union
            + self.w_edge * L_edge
        )

        loss_dict = {
            'L_total': total_loss.item(),
            'L_sup_full': L_sup_full.item(),
            'L_bce': L_bce.item(),
            'L_dice': L_dice.item(),
            'L_weak': L_weak.item(),
            'L_weak_vis': L_weak_vis.item(),
            'L_weak_occ': L_weak_occ.item(),
            'L_excl': L_excl.item(),
            'L_subset': L_subset.item(),
            'L_union': L_union.item(),
            'L_edge': L_edge.item(),
        }

        return total_loss, loss_dict


# ---- Quick test ----
if __name__ == '__main__':
    print("Testing TriBranchAmodalLoss...")

    B, H, W = 2, 160, 160
    pred_vis = torch.randn(B, 1, H, W)
    pred_occ = torch.randn(B, 1, H, W)
    pred_full = torch.randn(B, 1, H, W)
    amodal_gt = (torch.rand(B, 1, H, W) > 0.5).float()
    pseudo_vis = (torch.rand(B, 1, H, W) > 0.5).float()
    pseudo_occ = (torch.rand(B, 1, H, W) > 0.5).float()
    rgb_edges = torch.rand(B, 1, H, W)

    criterion = TriBranchAmodalLoss()
    total_loss, loss_dict = criterion(pred_vis, pred_occ, pred_full, amodal_gt, pseudo_vis, pseudo_occ, rgb_edges)

    print(f"Total loss: {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
    print("Loss test passed!")
