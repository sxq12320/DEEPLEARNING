# 西瓜花授粉点二阶段 ROI 热力图算法说明

## 1. 算法目标

当前方案采用“方向 A”：先用 YOLO 分割网络定位西瓜花区域，再把每个花朵实例裁剪成 ROI，输入二阶段关键点网络，直接预测授粉点热力图。最终输出授粉点坐标，并计算像素误差、OKS、mAP50 和 mAP50-95。

对应代码主要在：

- `E:\mastercode\ultralytics-main-new\013_improved_net_v2.py`
- `E:\mastercode\ultralytics-main-new\013_train_improved_v2.py`

## 2. 整体流程

1. YOLO 分割模型读取原图，输出每朵西瓜花的实例掩码。
2. 对每个掩码计算外接框，并按比例向外扩展，得到花朵 ROI。
3. 在原图中裁剪 ROI 图像，同时裁剪对应的二值 mask。
4. 将 ROI 图像缩放到 `128x128`，mask 也缩放到同样大小。
5. 构造 4 通道输入：`RGB + mask`。
6. 网络输出 `64x64` 的授粉点热力图。
7. 从热力图峰值解码出 ROI 内坐标，再映射回原图坐标。
8. 与人工标注 GT 比较，计算误差和关键点 mAP。

## 3. YOLO 分割阶段

默认分割模型路径为：

```text
E:\mastercode\ultralytics-main-new\results\09_watermelon_seg_2\weights\best.pt
```

这个模型需要提前训练好。二阶段网络不会重新训练 YOLO，它只是调用 YOLO 产生 mask。后续如果更换或改进 YOLO 分割网络，只要仍能输出西瓜花实例 mask，二阶段网络可以继续使用。

代码中会遍历 YOLO 输出的每一个 mask，而不是只取最大 mask。这样一张图中有多朵花时，每朵花都可以形成一个训练样本。

## 4. GT 匹配方式

每个 YOLO mask 先计算掩码质心，质心只用于匹配标注点，不再作为最终预测结果。程序会在对应 JSON 标注中寻找最近的点标注：

- 接受 `fully_visible` 和 `partially_visible`。
- 排除 `invisible`。
- 超过 `MAX_GT_MATCH_DISTANCE_PX = 160` 像素的点会被丢弃。

这样可以减少 YOLO mask 与错误 GT 点配对的问题。

## 5. ROI 样本构建

对每个有效 mask：

- 用 mask 外接矩形得到花朵区域。
- 默认按 `margin_ratio=0.25` 扩展边界，保留花瓣周边上下文。
- 生成 ROI RGB 图像和 ROI mask。
- 将 GT 授粉点从原图坐标转换为 ROI 内归一化坐标。
- 用 GT 坐标生成高斯热力图，默认 `sigma=2.0`。

训练样本实际包含：

```text
roi:        4 x 128 x 128
heatmap:    1 x 64 x 64
gt_roi_xy:  ROI 内归一化关键点坐标
gt_center:  原图归一化关键点坐标
roi_box:    原图中的 ROI 框
mask_area:  mask 面积，用于 OKS/mAP
```

## 6. 二阶段网络结构

`ROIHeatmapNet` 是一个轻量 U-Net 风格网络：

- 输入：4 通道 ROI，即 RGB 三通道加 mask 一通道。
- 编码端：多层卷积和下采样，提取花朵局部纹理和结构。
- 解码端：上采样并融合浅层特征。
- 输出：单通道授粉点 heatmap logits。

它不再只学习“64 个轮廓点质心到 GT 的偏差”，而是直接从 ROI 图像和 mask 中学习授粉点所在位置。这样网络可以利用颜色、纹理、花蕊形态、mask 几何结构等信息，表达能力更强。

## 7. 损失函数

训练损失由两部分组成：

```text
loss = heatmap_mse + 0.25 * coord_smooth_l1
```

其中：

- `heatmap_mse`：预测 heatmap 经过 sigmoid 后，与 GT 高斯热力图做 MSE。
- `coord_smooth_l1`：对预测 heatmap 做 soft-argmax，得到连续坐标，再与 GT ROI 坐标计算 SmoothL1。

这样既约束整张热力图形状，也约束最终坐标位置。

## 8. 指标计算

最终验证阶段会计算：

- `mean_error_px`：预测点和 GT 点的平均像素距离。
- `median_error_px`：像素误差中位数。
- `<10px / <20px / <30px`：不同误差阈值下的样本比例。
- `OKS`：用关键点距离、图像尺寸和 mask 面积归一化后的相似度。
- `mAP50`：OKS 大于等于 0.50 的比例。
- `mAP50-95`：OKS 阈值从 0.50 到 0.95，步长 0.05，取平均 AP。

这里的 mAP 是单关键点任务的 mAP，不是目标检测框 mAP。它评价的是授粉点定位是否足够接近 GT。

## 9. 训练命令

在 PowerShell 中运行：

```powershell
Set-Location E:\mastercode\ultralytics-main-new
python .\013_train_improved_v2.py --epochs 100 --batch-size 16
```

常用参数：

```powershell
python .\013_train_improved_v2.py `
  --epochs 100 `
  --batch-size 16 `
  --roi-size 128 `
  --heatmap-size 64 `
  --save-dir E:\mastercode\ultralytics-main-new\results\13_roi_heatmap_v2
```

调试小样本可运行：

```powershell
python .\013_train_improved_v2.py --epochs 1 --batch-size 2 --max-train-samples 2 --max-val-samples 2 --max-visualizations 4 --no-cache
```

## 10. 输出文件

默认输出目录：

```text
E:\mastercode\ultralytics-main-new\results\13_roi_heatmap_v2
```

主要结果：

- `best.pth`：按最佳 `mAP50-95` 保存的模型权重。
- `results.json`：最终指标、参数、mAP 结果和可视化目录。
- `training_curve.png`：训练损失、验证损失和 mAP 曲线。
- `visualizations\*.jpg`：GT 与 Pred 对比图。
- `visualizations\index.json`：每张可视化图对应的误差、GT 坐标、Pred 坐标和 ROI 框。

## 11. 可视化结果说明

训练结束后，程序会自动对验证集生成可视化图：

- 蓝色矩形：YOLO mask 得到的 ROI 区域。
- 绿色点：人工标注 GT 授粉点。
- 红色点：二阶段网络预测点。
- 黄色线：GT 与 Pred 之间的误差连线。
- 左上角文字：样本名和像素误差。

可视化图用于直观看模型是否预测在花蕊区域、是否受到错误 mask 影响、是否存在系统性偏移。

## 12. 推荐改进方向

后续优先改进 YOLO 分割质量，因为二阶段网络的输入 ROI 依赖 YOLO mask。如果 mask 漏检、错检或边界严重偏移，二阶段关键点网络也会受影响。

其次可以改进二阶段网络：

- 增加数据增强，如亮度、对比度、旋转、轻微缩放。
- 将 `sigma` 调成 1.5 到 3.0 之间做对比实验。
- 对困难样本增加权重，例如误差较大或花蕊遮挡样本。
- 尝试更强的 backbone，但要注意数据量不足时可能过拟合。
- 将 ROI heatmap 可视化也保存出来，用于判断网络是否学到稳定峰值。

当前方向比“64 个点求质心再回归偏差”更合理，因为最终任务本质是关键点定位，而 heatmap 方法能直接学习空间概率分布，也更容易用 mAP、OKS 和可视化结果进行评估。
