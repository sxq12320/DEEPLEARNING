"""
Amodal RGB-D Dataset with On-the-fly Pseudo Label Generation.

Supports two data formats:
  1. 'npy' format (default): 4-channel .npy files (BGRD) + YOLO polygon labels
     - images/: .npy files with shape (H, W, 4) — B, G, R, Depth(float32)
     - labels/: .txt files with YOLO polygon format (class x1 y1 x2 y2 ... xn yn)

  2. 'separate' format: separate RGB, Depth, and Amodal mask directories
     - rgb/:    RGB images (.png/.jpg)
     - depth/:  Depth maps (.png/.npy, single channel)
     - amodal/: Amodal GT masks (.png, binary)

On-the-fly operations (no offline preprocessing):
  - Pseudo visible/occluded mask generation from depth
  - RGB edge extraction via Canny
"""

import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path


class AmodalRGBDDataset(Dataset):
    """RGB-D Amodal Segmentation Dataset with on-the-fly pseudo label generation."""

    def __init__(
        self,
        root,
        split='train',
        img_size=640,
        data_format='npy',
        depth_sobel_thresh=0.1,
        canny_thresh1=100,
        canny_thresh2=200,
        dilate_kernel_size=5,
        depth_norm_clip=10.0,
        augment=False,
    ):
        """
        Args:
            root: dataset root directory
            split: 'train' or 'val'
            img_size: target image size (square)
            data_format: 'npy' (4-channel numpy + polygon labels) or 'separate' (separate dirs)
            depth_sobel_thresh: threshold for depth Sobel gradient to detect occlusion boundaries
            canny_thresh1: Canny lower threshold for RGB edge extraction
            canny_thresh2: Canny upper threshold for RGB edge extraction
            dilate_kernel_size: kernel size for morphological dilation of occlusion boundaries
            depth_norm_clip: clip depth values to [0, depth_norm_clip] before normalization
            augment: whether to apply data augmentation (horizontal flip only for now)
        """
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.data_format = data_format
        self.depth_sobel_thresh = depth_sobel_thresh
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2
        self.dilate_kernel_size = dilate_kernel_size
        self.depth_norm_clip = depth_norm_clip
        self.augment = augment and (split == 'train')

        # Build file list
        self.samples = self._build_file_list()

        # Dilation kernel for occlusion boundary expansion
        self.dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size)
        )

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.root}/{self.split} with format '{data_format}'")
        print(f"[AmodalRGBDDataset] {split}: {len(self.samples)} samples, format={data_format}, img_size={img_size}")

    def _build_file_list(self):
        """Build list of (image_path, label/mask_path) tuples."""
        samples = []
        split_dir = self.root / self.split

        if self.data_format == 'npy':
            img_dir = split_dir / 'images'
            lbl_dir = split_dir / 'labels'
            if not img_dir.exists():
                raise FileNotFoundError(f"Image directory not found: {img_dir}")
            for img_path in sorted(img_dir.glob('*.npy')):
                lbl_path = lbl_dir / (img_path.stem + '.txt')
                if lbl_path.exists():
                    samples.append({'image': str(img_path), 'label': str(lbl_path)})
                else:
                    # Allow samples without labels (will produce empty masks)
                    samples.append({'image': str(img_path), 'label': None})

        elif self.data_format == 'separate':
            rgb_dir = split_dir / 'rgb'
            depth_dir = split_dir / 'depth'
            amodal_dir = split_dir / 'amodal'
            if not rgb_dir.exists():
                raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
            for rgb_path in sorted(rgb_dir.iterdir()):
                stem = rgb_path.stem
                # Try common depth extensions
                depth_path = None
                for ext in ['.png', '.npy', '.jpg', '.tif']:
                    candidate = depth_dir / (stem + ext)
                    if candidate.exists():
                        depth_path = str(candidate)
                        break
                # Try common amodal extensions
                amodal_path = None
                for ext in ['.png', '.npy', '.jpg', '.tif']:
                    candidate = amodal_dir / (stem + ext)
                    if candidate.exists():
                        amodal_path = str(candidate)
                        break
                if depth_path and amodal_path:
                    samples.append({
                        'rgb': str(rgb_path),
                        'depth': depth_path,
                        'amodal': amodal_path,
                    })
        else:
            raise ValueError(f"Unknown data_format: {self.data_format}. Use 'npy' or 'separate'.")

        return samples

    def _load_npy_sample(self, sample):
        """Load sample in 'npy' format: 4-channel .npy + polygon label."""
        rgbd = np.load(sample['image']).astype(np.float32)  # (H, W, 4)

        # Extract RGB (BGR -> RGB) and Depth
        rgb = rgbd[:, :, :3][:, :, ::-1].copy()  # BGR to RGB
        depth = rgbd[:, :, 3:4]                    # (H, W, 1)

        # Convert polygon label to binary mask
        h, w = rgb.shape[:2]
        amodal_mask = np.zeros((h, w), dtype=np.float32)
        if sample['label'] is not None:
            amodal_mask = self._polygon_to_mask(sample['label'], h, w)

        return rgb, depth, amodal_mask

    def _load_separate_sample(self, sample):
        """Load sample in 'separate' format: individual RGB, Depth, Amodal files."""
        # RGB
        rgb = cv2.imread(sample['rgb'])  # BGR
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Depth
        depth_path = sample['depth']
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path).astype(np.float32)
            if depth.ndim == 2:
                depth = depth[:, :, np.newaxis]
        else:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            if depth.ndim == 2:
                depth = depth[:, :, np.newaxis]

        # Amodal mask
        amodal_path = sample['amodal']
        if amodal_path.endswith('.npy'):
            amodal_mask = np.load(amodal_path).astype(np.float32)
        else:
            amodal_mask = cv2.imread(amodal_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            amodal_mask = (amodal_mask > 127).astype(np.float32)

        return rgb, depth, amodal_mask

    @staticmethod
    def _polygon_to_mask(label_path, height, width):
        """Convert YOLO polygon label file to binary mask.

        Label format: each line is "class_id x1 y1 x2 y2 ... xn yn"
        where coordinates are normalized to [0, 1].
        """
        mask = np.zeros((height, width), dtype=np.float32)
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:  # class + at least 3 points (6 coords)
                    continue
                coords = [float(x) for x in parts[1:]]
                pts = np.array(coords).reshape(-1, 2)
                pts = (pts * [width, height]).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1.0)
        return mask

    def _resize_all(self, rgb, depth, amodal_mask):
        """Resize all inputs to target size."""
        h, w = self.img_size, self.img_size
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        # Depth uses nearest to avoid introducing invalid values
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        if amodal_mask.shape[:2] != (h, w):
            amodal_mask = cv2.resize(amodal_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return rgb, depth, amodal_mask

    def _generate_pseudo_labels(self, depth, amodal_mask):
        """Generate pseudo visible and occluded masks from depth map.

        Strategy:
          1. Depth black-hole (value=0) within Amodal_GT → Occluded
          2. Depth Sobel gradient > threshold within Amodal_GT → Occlusion boundary,
             dilate and add to Occluded
          3. pseudo_vis = Amodal_GT - pseudo_occ

        Args:
            depth: (H, W, 1) float32 depth map (raw values, before normalization)
            amodal_mask: (H, W) float32 binary amodal GT mask

        Returns:
            pseudo_vis: (H, W) float32 binary visible mask
            pseudo_occ: (H, W) float32 binary occluded mask
        """
        depth_2d = depth[:, :, 0]  # (H, W)
        amodal_binary = (amodal_mask > 0.5)

        # --- Step 1: Depth black-hole occlusion ---
        depth_hole = (depth_2d == 0) | np.isnan(depth_2d)
        occ_from_hole = (depth_hole & amodal_binary).astype(np.float32)

        # --- Step 2: Depth Sobel gradient occlusion boundary ---
        # Normalize depth for gradient computation
        depth_norm = np.clip(depth_2d, 0, self.depth_norm_clip)
        valid_mask = ~depth_hole
        if valid_mask.any():
            d_min = depth_norm[valid_mask].min()
            d_max = depth_norm[valid_mask].max()
            if d_max - d_min > 1e-6:
                depth_for_grad = (depth_norm - d_min) / (d_max - d_min)
            else:
                depth_for_grad = np.zeros_like(depth_norm)
        else:
            depth_for_grad = np.zeros_like(depth_norm)

        # Set holes to 0 for gradient computation
        depth_for_grad = depth_for_grad * (~depth_hole).astype(np.float32)
        depth_for_grad = (depth_for_grad * 255).astype(np.uint8)  # cv2.Sobel needs uint8

        # Sobel gradients
        sobel_x = cv2.Sobel(depth_for_grad, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(depth_for_grad, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # Normalize gradient to [0, 1]
        if gradient_mag.max() > 0:
            gradient_mag = gradient_mag / gradient_mag.max()

        # Threshold gradient within amodal region
        gradient_occ = ((gradient_mag > self.depth_sobel_thresh) & amodal_binary).astype(np.float32)

        # Dilate occlusion boundaries
        gradient_occ = cv2.dilate(gradient_occ, self.dilate_kernel, iterations=1)

        # --- Combine occlusion masks ---
        pseudo_occ = np.maximum(occ_from_hole, gradient_occ)
        # Ensure occ is within amodal region
        pseudo_occ = pseudo_occ * amodal_binary.astype(np.float32)

        # --- Visible = Amodal - Occluded ---
        pseudo_vis = amodal_mask - pseudo_occ
        pseudo_vis = np.clip(pseudo_vis, 0, 1)

        return pseudo_vis, pseudo_occ

    def _extract_rgb_edges(self, rgb):
        """Extract edges from RGB image using Canny detector.

        Args:
            rgb: (H, W, 3) uint8 RGB image

        Returns:
            edges: (H, W) float32 edge map normalized to [0, 1]
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self.canny_thresh1, self.canny_thresh2)
        edges = edges.astype(np.float32) / 255.0
        return edges

    def _apply_augmentation(self, rgb, depth, amodal_mask, pseudo_vis, pseudo_occ, rgb_edges):
        """Apply random horizontal flip augmentation."""
        if np.random.random() > 0.5:
            rgb = np.flip(rgb, axis=1).copy()
            depth = np.flip(depth, axis=1).copy()
            amodal_mask = np.flip(amodal_mask, axis=1).copy()
            pseudo_vis = np.flip(pseudo_vis, axis=1).copy()
            pseudo_occ = np.flip(pseudo_occ, axis=1).copy()
            rgb_edges = np.flip(rgb_edges, axis=1).copy()
        return rgb, depth, amodal_mask, pseudo_vis, pseudo_occ, rgb_edges

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # ---- Load data ----
        if self.data_format == 'npy':
            rgb, depth, amodal_mask = self._load_npy_sample(sample)
        else:
            rgb, depth, amodal_mask = self._load_separate_sample(sample)

        # ---- Resize ----
        rgb, depth, amodal_mask = self._resize_all(rgb, depth, amodal_mask)

        # ---- On-the-fly pseudo label generation ----
        pseudo_vis, pseudo_occ = self._generate_pseudo_labels(depth, amodal_mask)

        # ---- On-the-fly RGB edge extraction ----
        rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb_edges = self._extract_rgb_edges(rgb_uint8)

        # ---- Augmentation ----
        if self.augment:
            rgb, depth, amodal_mask, pseudo_vis, pseudo_occ, rgb_edges = self._apply_augmentation(
                rgb, depth, amodal_mask, pseudo_vis, pseudo_occ, rgb_edges
            )

        # ---- Normalize and convert to tensors ----
        # RGB: [0, 255] -> [0, 1], then ImageNet normalize
        rgb_tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)  # [3, H, W]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        rgb_tensor = (rgb_tensor - mean) / std

        # Depth: clip and normalize to [0, 1]
        depth_float = np.clip(depth, 0, self.depth_norm_clip) / self.depth_norm_clip
        depth_tensor = torch.from_numpy(depth_float.astype(np.float32)).permute(2, 0, 1)  # [1, H, W]

        # Concat RGB + Depth -> RGBD [4, H, W]
        rgbd_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)  # [4, H, W]

        # Masks: [1, H, W]
        amodal_tensor = torch.from_numpy(amodal_mask.astype(np.float32)).unsqueeze(0)  # [1, H, W]
        vis_tensor = torch.from_numpy(pseudo_vis.astype(np.float32)).unsqueeze(0)       # [1, H, W]
        occ_tensor = torch.from_numpy(pseudo_occ.astype(np.float32)).unsqueeze(0)       # [1, H, W]
        edge_tensor = torch.from_numpy(rgb_edges.astype(np.float32)).unsqueeze(0)       # [1, H, W]

        return {
            'rgbd': rgbd_tensor,         # [4, H, W]
            'amodal_gt': amodal_tensor,   # [1, H, W]
            'pseudo_vis': vis_tensor,     # [1, H, W]
            'pseudo_occ': occ_tensor,     # [1, H, W]
            'rgb_edges': edge_tensor,     # [1, H, W]
        }


# ---- Quick test ----
if __name__ == '__main__':
    import sys

    # Test with synthetic data
    print("Testing AmodalRGBDDataset with synthetic data...")

    # Create a temporary test directory
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic 'npy' format data
        split_dir = os.path.join(tmpdir, 'train')
        img_dir = os.path.join(split_dir, 'images')
        lbl_dir = os.path.join(split_dir, 'labels')
        os.makedirs(img_dir)
        os.makedirs(lbl_dir)

        # Create a synthetic 4-channel numpy file
        h, w = 480, 640
        rgbd = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8).astype(np.float32)
        depth = np.random.rand(h, w, 1).astype(np.float32) * 5.0
        # Add some zero-depth regions (occluded)
        depth[100:200, 200:300, 0] = 0.0
        rgbd_full = np.concatenate([rgbd[:, :, ::-1], depth], axis=2)  # BGRD

        np.save(os.path.join(img_dir, 'test_001.npy'), rgbd_full)

        # Create a polygon label
        with open(os.path.join(lbl_dir, 'test_001.txt'), 'w') as f:
            # A rectangular polygon covering the occluded region and more
            f.write("0 0.2 0.2 0.6 0.2 0.6 0.6 0.2 0.6\n")

        # Test dataset
        dataset = AmodalRGBDDataset(
            root=tmpdir,
            split='train',
            img_size=640,
            data_format='npy',
        )

        sample = dataset[0]
        print(f"rgbd:       {sample['rgbd'].shape}, range=[{sample['rgbd'].min():.3f}, {sample['rgbd'].max():.3f}]")
        print(f"amodal_gt:  {sample['amodal_gt'].shape}, unique={torch.unique(sample['amodal_gt'])}")
        print(f"pseudo_vis: {sample['pseudo_vis'].shape}, unique={torch.unique(sample['pseudo_vis'])}")
        print(f"pseudo_occ: {sample['pseudo_occ'].shape}, unique={torch.unique(sample['pseudo_occ'])}")
        print(f"rgb_edges:  {sample['rgb_edges'].shape}, range=[{sample['rgb_edges'].min():.3f}, {sample['rgb_edges'].max():.3f}]")
        print("Dataset test passed!")
