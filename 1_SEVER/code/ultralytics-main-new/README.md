# YOLO11-RGBD: Apple Amodal Detection

基于 Ultralytics YOLO11 的 RGB-D 苹果遮挡检测模型。采用纯 CNN + 频域 + 动态门控机制，不依赖 Transformer 或 Mamba。

## 自定义模块

### SFM (Strip-Freq Mixer)

**用途**: 替换 Backbone 中的 `C3k2` / `C2f`

SFM 采用双分支并行架构，融合条带感知与全局频域建模：

| 分支 | 机制 | 说明 |
|------|------|------|
| **Branch A** (条带感知) | 正交条形深度卷积 (`1×K` + `K×1`) | 捕获长程依赖，匹配果园枝条/叶片形状 |
| **Branch B** (全局频域) | 2D-FFT → 实部/虚部分离 → Conv → IFFT | 全局上下文建模 |

**融合方式**: Concat(A, B) → 1×1 Conv + 残差连接

**FFT 安全处理流程**:
1. `rfft2` 获取频域表示
2. 分离实部/虚部，沿通道维度拼接
3. 通过标准 `nn.Conv2d` 处理
4. 重组复数张量，`irfft2` 还原空间域

---

### WCAF (Wavelet-Cross-Attention Fusion)

**用途**: 替换 Neck 中 RGB 与 Depth 特征交汇处的 `Concat`

WCAF 利用深度信息的几何先验来抑制 RGB 光照/阴影噪声：

1. 对 RGB 和 Depth 特征分别进行 2D Haar 小波变换 (DWT)
2. 用 Depth 的低频子带 (LL) 生成空间注意力图 (`1×1 Conv` + `Sigmoid`)
3. 用该注意力图对 RGB 的高频子带 (LH, HL, HH) 进行门控
4. 逆小波变换 (IDWT) 重建增强后的特征

**小波实现**: 手写 Haar DWT/IDWT（纯 PyTorch 张量切片），无外部依赖，兼容 ONNX 导出。

---

### DGFFN (Dilated-Gated FFN)

**用途**: 替换标准 YOLO FFN（两个 1×1 Conv）

DGFFN 通过多尺度膨胀卷积 + 通道注意力 + GLU 门控增强特征表达：

1. **1×1 Conv** 通道扩展
2. **多尺度膨胀 DWConv**: 通道对半拆分，分别使用 `3×3 DWConv (dilation=1)` 和 `5×5 DWConv (dilation=2)`
3. **通道注意力 (CA)**: 全局平均池化 → 1×1 Conv → Sigmoid → 逐通道加权
4. **GLU (门控线性单元)**: 通道对半拆分，一半乘以另一半的 Sigmoid
5. **1×1 Conv** 通道投影 + 残差连接

---

## 文件结构

```
ultralytics/
├── nn/
│   └── modules/
│       ├── custom_blocks.py    # SFM, WCAF, DGFFN, HaarDWT, HaarIDWT
│       ├── __init__.py         # 模块导出
│       └── ...
│   └── tasks.py                # parse_model 解析逻辑
├── cfg/
│   └── models/
│       └── 11/
│           └── yolo11-rgbd.yaml  # 模型配置文件
└── data/
    └── base.py                 # 4通道 (RGBD) 图像读取支持
```

## 模型配置

YAML 配置文件: `ultralytics/cfg/models/11/yolo11-rgbd.yaml`

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]        # P1/2
  - [-1, 1, Conv, [128, 3, 2]]       # P2/4
  - [-1, 2, SFM, [256]]              # SFM 替换 C3k2
  - [-1, 1, Conv, [256, 3, 2]]       # P3/8
  - [-1, 2, SFM, [512]]              # SFM 替换 C3k2
  # ...

neck:
  - [[-1, 6], 1, WCAF, []]           # WCAF 替换 Concat (RGB+Depth 融合)
  - [-1, 2, DGFFN, [512]]            # DGFFN 替换 C3k2 FFN
  # ...
```

## 使用方法

### CLI

```bash
yolo task=detect mode=train model=yolo11-rgbd.yaml data=your_dataset.yaml epochs=100 imgsz=640
```

### Python

```python
from ultralytics import YOLO

model = YOLO("yolo11-rgbd.yaml")
model.info()
results = model.train(data="your_dataset.yaml", epochs=100, imgsz=640)
```

## 模型规模

| 规模 | depth | width | max_channels | 参数量 (约) |
|------|-------|-------|-------------|------------|
| n    | 0.50  | 0.25  | 1024        | 3.4M       |
| s    | 0.50  | 0.50  | 1024        | -          |
| m    | 0.50  | 1.00  | 512         | -          |
| l    | 1.00  | 1.00  | 512         | -          |
| x    | 1.00  | 1.50  | 512         | -          |

## 设计约束

- **禁止** Transformer / Mamba 架构
- 纯 CNN + 频域 (FFT/小波) + 动态门控
- FFT 复数张量安全处理：实部/虚部分离后通过标准 Conv2d
- 小波变换：手写 Haar DWT/IDWT，无外部依赖，兼容 ONNX 导出
- 输入通道: 4 (RGB-D)
