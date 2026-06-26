# Segmentation-Guided Pollination Keypoint Methods

Date: 2026-06-25

Question: keep the first-stage flower segmentation, but look for better second-stage pollination-point localization methods than the current custom ROI heatmap U-Net.

## Current Local Baseline

- Local document: `3_研究生/西瓜花授粉点二阶段ROI热力图算法说明.md`
- Local code:
  - `ultralytics-main-new/013_improved_net_v2.py`
  - `ultralytics-main-new/013_train_improved_v2.py`
- Pipeline:
  - Stage 1: YOLO instance segmentation generates one mask per flower.
  - Stage 2: crop ROI from mask, resize to 128x128, concatenate RGB + binary mask, predict one 64x64 heatmap.
  - Training loss: heatmap MSE plus SmoothL1 on soft-argmax coordinates.

Main concerns found from local code:

- A 64x64 heatmap for a 128x128 ROI can introduce quantization and decode bias.
- The second-stage network is trained from scratch and does not use modern pose-estimation backbones or pretrained representations.
- GT matching is mask-centroid nearest-neighbor based and should be made one-to-one and visibility-aware.
- Training samples are built from predicted YOLO masks, which mixes first-stage errors into second-stage learning. This is useful for robustness, but should be separated from a cleaner GT-ROI training mode for diagnosis.
- The code currently uses RGB + mask; if the project is truly RGB-D, aligned depth is not yet being used in this second stage.

## Better Candidate Routes

### 1. Recommended: YOLO-Seg + Top-Down Keypoint Estimator

Keep YOLO segmentation as stage 1. Use each mask to form an ROI box and optional mask prior, then train a standard top-down pose/keypoint estimator with one keypoint: the pollination point.

Good second-stage choices:

- RTMPose/SimCC style head for low-resolution and fast inference.
- Lite-HRNet or HRNet heatmap head with UDP/DARK decoding for stronger spatial precision.
- ResNet + deconv heatmap head as a simple, reliable baseline.

Why this is likely better:

- Top-down pose estimation is the standard pattern: first detect/propose an instance, then localize keypoints inside the crop.
- HRNet-style backbones preserve high-resolution feature maps, which is important for a small point on a flower.
- SimCC reformulates x/y localization as 1D coordinate classification and directly targets low quantization error.
- UDP and DARK address coordinate transform and heatmap decoding bias without changing the first-stage segmentation.

Implementation outline:

- Convert labels to COCO-style keypoint annotations with one keypoint and visibility.
- For training, use GT masks/boxes first, then fine-tune/evaluate using YOLO segmentation boxes/masks.
- At inference, run YOLO segmentation, pad each mask bounding box, optionally apply mask as input prior, and pass the crop or bbox to the top-down model.
- Keep evaluation identical to the current script: mean/median pixel error, OKS, mAP50, mAP50-95, plus PCK-style thresholds if useful.

### 2. Keep Current Code But Fix the Second-Stage Formulation

This is lower engineering cost and should be used as the immediate ablation baseline.

Changes to test:

- Increase heatmap from 64x64 to 128x128, or use a super-resolution/high-resolution head.
- Replace plain argmax decoding with DARK or UDP-style unbiased decoding.
- Use integral regression or SimCC-style 1D coordinate distributions instead of only 2D heatmap argmax.
- Fix GT assignment:
  - exclude invisible labels before nearest-neighbor selection;
  - one-to-one match masks and GT points, preferably Hungarian assignment;
  - require the GT point to be inside or near the mask/expanded mask;
  - suppress duplicate masks matched to the same GT.
- Train with crop jitter, scale jitter, rotation, color/illumination augmentation, random mask erosion/dilation, and synthetic first-stage box/mask perturbation.
- Report separate performance using GT ROI and YOLO-predicted ROI to separate second-stage localization error from first-stage segmentation error.

### 3. Larger Rewrite: Mask/Keypoint R-CNN-Style ROIAlign Head

This also keeps segmentation, but turns the problem into a unified instance model:

- shared backbone extracts image features;
- mask branch predicts flower segmentation;
- keypoint branch uses ROIAlign features to predict the pollination point.

Why it may help:

- ROIAlign reduces crop/resize alignment problems.
- Segmentation and keypoint heads can share features.
- It is a mature architecture for instance segmentation plus keypoints.

Why it is not the first recommendation here:

- It is a bigger rewrite from the current Ultralytics YOLO fork.
- With a small custom agricultural dataset, a separate top-down model is easier to debug and compare.
- If YOLO segmentation is already strong and must remain, replacing only stage 2 gives faster feedback.

### 4. RGB-D / 3D-Aware Extension

If aligned depth is available, it should be tested explicitly:

- Add depth as a fifth channel: RGB + mask + depth.
- Or crop a flower point cloud from the stage-1 mask and regress 2D point plus local surface normal or 3D contact point.
- Pollination literature increasingly treats flower pose as a 3D pose problem, not only a 2D pixel point problem.

This is valuable if the robot/end effector needs a physical contact point, but it should follow a solid 2D baseline.

## Practical Recommendation

Run these ablations in order:

1. Fix current data matching and label handling, then re-run the current baseline.
2. Current model + 128x128 heatmap and DARK/UDP-style decoding.
3. YOLO-Seg + MMPose top-down one-keypoint RTMPose/SimCC model.
4. YOLO-Seg + Lite-HRNet or HRNet heatmap+UDP model.
5. Optional: ROIAlign keypoint head or Keypoint R-CNN-style integration if route 3/4 plateaus.
6. Optional: RGB-D or synthetic-data/teacher-student route for robotic deployment.

Expected best engineering tradeoff:

YOLO segmentation remains stage 1. Replace the custom second-stage U-Net with a top-down one-keypoint pose estimator using SimCC or UDP/DARK heatmap decoding. This directly attacks the current method's main weaknesses while preserving the segmentation-first pipeline.

## Sources

- Simple Baselines for Human Pose Estimation and Tracking. arXiv: https://arxiv.org/abs/1804.06208
- Deep High-Resolution Representation Learning for Human Pose Estimation. arXiv: https://arxiv.org/abs/1902.09212
- HRNet official implementation: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
- Lite-HRNet: A Lightweight High-Resolution Network. arXiv: https://arxiv.org/abs/2104.06403
- Distribution-Aware Coordinate Representation for Human Pose Estimation (DARK). arXiv: https://arxiv.org/abs/1910.06278
- The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation (UDP). arXiv: https://arxiv.org/abs/1911.07524
- SimCC: a Simple Coordinate Classification Perspective for Human Pose Estimation. arXiv: https://arxiv.org/abs/2107.03332
- Integral Human Pose Regression. arXiv: https://arxiv.org/abs/1711.08229
- Mask R-CNN. arXiv: https://arxiv.org/abs/1703.06870
- Torchvision Keypoint R-CNN docs: https://docs.pytorch.org/vision/main/models/keypoint_rcnn.html
- MMPose codecs documentation: https://mmpose.readthedocs.io/en/latest/advanced_guides/codecs.html
- MMPose top-down API: https://mmpose.readthedocs.io/en/latest/api.html
- MMPose model zoo: https://mmpose.readthedocs.io/en/latest/model_zoo.html
- RTMPose. arXiv: https://arxiv.org/html/2303.07399v2
- ViTPose. arXiv: https://arxiv.org/abs/2204.12484
- Ultralytics pose task docs: https://docs.ultralytics.com/tasks/pose/
- FloPE: Flower Pose Estimation for Precision Pollination. arXiv: https://arxiv.org/abs/2503.11692
- FloPE official repository: https://github.com/wvu-irl/flope
- Robotic 3D Flower Pose Estimation for Small-Scale Urban Farms. arXiv: https://arxiv.org/abs/2509.02870
