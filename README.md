mastercode
深度学习计算机视觉研究与实践仓库，涵盖图像分割、目标检测（RGB-D 苹果遮挡检测）与经典网络架构学习三大方向。

仓库结构

Plain Text

.
├── 1.coding/                 # 深度学习编码项目
│   ├── 0_segment/            # 图像分割与检测工程骨架（模块化 + 注册机制 + 配置驱动）
│   ├── 1_study_module/       # 经典网络架构复现（LeNet → ConvNeXt / LSNet）
│   ├── 2_Unet/               # U-Net 分割实现
│   └── 3_phics_x/            # PHICS-X 模型（ResNet18 基座）
├── 2_catoon/                 # Manim 数学动画（神经网络可视化）
├── ultralytics-main-new/     # 定制版 Ultralytics YOLO11（RGB-D 苹果遮挡检测）
└── .github/                  # CI：Issue 触发自动更新本 README 博客列表
1. 1.coding — 深度学习编码项目
0_segment — 图像分割与检测工程骨架
基于 PyTorch 的模块化工程，支持图像分割与目标检测两大任务。采用注册机制 + 配置驱动组网，可灵活更换 backbone / neck / head。

分割模型：MiniSegNet（ResNet18 + 1×1 Conv）、FPNSegNet（多尺度 ResNet18 + FPN）
检测模型：YOLO11Detector（CSP + SPPF 骨干 + PAN-FPN 颈部 + 解耦头）
多模态：TS-Dual（RGB + Mask 先验 + Depth，输出分割与边界框）
数据格式：mask 图片 / YOLO TXT / COCO JSON / NumPy NPY
损失函数：分割 BCE/CE；检测 CIoU + DFL + BCE（TaskAlignedAssigner 标签分配）
详见 1.coding/0_segment/README.md。


Bash

cd 1.coding/0_segment
pip install -r requirements.txt
python train.py --model-type fpnseg          # 分割
python train.py --model-type ts_dual ...     # 多模态
1_study_module — 经典网络架构复现
按论文顺序复现主流网络，每个子目录对应一个架构：

编号	架构	编号	架构
1	LeNet-5	9	NIN (Network in Network)
2	AlexNet	10	GoogleNet
3	VGGNet (A~E)	11	ResNet
4	SEBlock	12	DenseNet
5	CBAM	13	ConvNeXt
6	MobileNet-V1	14	MobileNet-V2
7	Transformer	15	FCN (全卷积网络)
8	ViT	16	LSNet
2_Unet / 3_phics_x
2_Unet — U-Net 医学图像分割实现（含训练图像样本）
3_phics_x — PHICS-X 模型，基于 ResNet18 基座
2. 2_catoon — Manim 数学动画
使用 Manim 制作神经网络原理动画，用于可视化教学（如 LeNet 结构演示）。

3. ultralytics-main-new — RGB-D 苹果遮挡检测
基于 Ultralytics YOLO11 的定制版本，采用纯 CNN + 频域 + 动态门控机制（不依赖 Transformer / Mamba），面向果园 RGB-D 苹果遮挡检测。

自定义模块
模块	作用	机制
SFM (Strip-Freq Mixer)	替换 backbone 的 C3k2/C2f	条带感知深度卷积 + 2D-FFT 全局频域建模
WCAF (Wavelet-Cross-Attention Fusion)	替换 neck 的 Concat	Haar 小波变换，Depth 低频门控 RGB 高频
DGFFN (Dilated-Gated FFN)	替换标准 FFN	多尺度膨胀卷积 + 通道注意力 + GLU 门控
异构双流骨干
RGB 分支：MobileNetV3-Large / MobileNetV4（倒残差 + SE + HardSwish）
Depth 分支：StarNet（星型操作）/ ShuffleNetV2
尺度感知融合：P3 Depth2RGB、P4 双向互补、P5 RGB-led 几何先验
优化器：PIDAO（多通道高阶 PID 优化器）
实验配置
mine_yaml/ — 11 组消融与对比实验（baseline → 完整模型）
mine_yaml_v4/ — V4 版本 10 组实验
results/ — 训练结果（曲线、混淆矩阵、预测样例）
详见 ultralytics-main-new/README.md。


Bash

cd ultralytics-main-new
yolo task=detect mode=train model=yolo11-rgbd.yaml data=your_dataset.yaml epochs=100 imgsz=640
自动化：Issue 驱动博客列表
仓库配置了 GitHub Actions（.github/workflows/main.yml）：当带有 blog 标签的 Issue 被创建/编辑时，自动将标题与链接追加到下方列表。

技术栈
框架：PyTorch、Ultralytics YOLO11
领域：图像分割、目标检测、RGB-D 多模态融合、频域/小波变换
工具：Manim（动画）、GitHub Actions（CI）
语言：Python