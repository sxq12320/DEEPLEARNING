# 深度学习与计算机视觉综合代码库

欢迎来到本代码仓库！这是一个包含多个深度学习、图像分割、目标检测算法实现以及神经网络原理可视化的综合性工程集合。仓库代码主要涵盖了从经典网络学习、模块化分割框架搭建到前沿 RGB-D 目标检测模型开发的完整链路。

## 📁 仓库结构

仓库主要由以下三个核心模块组成：

### 1. `1.coding/` - 图像分割与经典网络学习
这是一个深度学习图像分割和经典神经网络学习的项目集合，适合学习与科研实验。
- **`0_segment/`**: 图像分割工程骨架。支持模块化设计、动态组网，通过配置文件即可快速构建基于 ResNet 等骨干网络的分割模型，支持多种数据标签格式。
- **`1_study_module/`**: 深度学习经典网络学习模块。包含了计算机视觉领域众多经典模型的纯手写实现与测试，包括：
  - **分类网络**: LeNet, AlexNet, VGGNet, GoogLeNet, ResNet, DenseNet, NIN, MobileNet (V1/V2), ConvNeXt 等。
  - **注意力机制与 Transformer**: SEBlock, CBAM, Transformer, ViT 等。
  - **分割与其他**: FCN, LSNET 等。
- **`2_Unet/`**: U-Net 图像分割网络的标准实现与训练流程。
- **`3_phics_x/`**: PHICS-X 相关模型与模块实现。

### 2. `2_catoon/` - 神经网络动画可视化
基于 [Manim](https://github.com/3b1b/manim) 引擎的数学与神经网络可视化动画项目。
- 包含 `0_Learning` 基础学习测试脚本和 `1_LeNet` 经典网络的结构展示（如 `abstract`, `detail`, `conclusion` 等分镜脚本）。
- 用于生成高质量的深度学习原理解析视频与动画，帮助更直观地理解网络的前向传播与结构。

### 3. `ultralytics-main-new/` - YOLO11-RGBD 苹果遮挡检测模型
基于官方 [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO 的深度定制版本，专为 RGB-D (4通道) 图像的苹果遮挡检测任务设计。
- **核心特点**: 遵循“纯 CNN + 频域 (FFT/小波) + 动态门控”的设计哲学，**完全不依赖 Transformer 或 Mamba**。
- **自定义创新模块**:
  - **SFM (Strip-Freq Mixer)**: 结合正交条形深度卷积与 2D-FFT 的全局频域建模，替换原始的 C3k2/C2f 模块。
  - **WCAF (Wavelet-Cross-Attention Fusion)**: 利用手写 Haar 小波变换 (DWT/IDWT) 处理 RGB 与 Depth 特征，通过深度空间注意力对 RGB 高频噪声进行门控过滤。
  - **DGFFN (Dilated-Gated FFN)**: 结合多尺度膨胀卷积、通道注意力与 GLU 门控，替换标准 YOLO 的前馈网络。
- **实验与配置**: 包含大量定制化的 YAML 模型配置（位于 `mine_yaml/` 与 `mine_yaml_v4/`）以及消融实验的详细结果（`results/`）。

---

## 🚀 快速开始

各模块均可独立运行，请进入对应目录查看详细说明或执行代码：

### 图像分割框架 (`1.coding/0_segment`)
```bash
cd 1.coding/0_segment
pip install -r requirements.txt
python scripts/train.py --epochs 10 --batch 8
```
> 详情请参考 `1.coding/0_segment/README.md`

### YOLO11-RGBD 目标检测 (`ultralytics-main-new`)
```bash
cd ultralytics-main-new
pip install -e .
yolo task=detect mode=train model=cfg/models/11/yolo11-rgbd.yaml data=your_dataset.yaml epochs=100 imgsz=640
```
> 模型设计细节与频域/小波模块说明请参考 `ultralytics-main-new/README.md`

### 动画渲染 (`2_catoon`)
需提前配置好 Manim 环境及 FFmpeg，然后执行对应的 Python 脚本即可渲染视频：
```bash
cd 2_catoon/1_LeNet
manim -pql 1_abstract.py SimpleCircle
```

## 📝 许可协议
本项目包含多个子模块，大部分基于开源社区项目二次开发。其中 YOLO 相关代码遵循 Ultralytics 原有的 AGPL-3.0 许可证，其余代码请参考各子目录下的具体说明。
