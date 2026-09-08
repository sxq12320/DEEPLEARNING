# E08：基于训练集 RGB 热图的有目的切片

## 动机

固定四角切片能放大极小果实，但它并不知道果实在哪里，因而可能把计算预算放在叶片或重复覆盖区域。E08 将“候选区域定位”和“分割器训练”分开：先用训练集标注训练一个极小的 RGB 热图引导器，再冻结引导器，用热图决定局部窗口位置。验证集和测试集不参与引导器训练，也不把真实标注传给切片器。

## 证据依据与边界

设计借鉴了 QueryDet 的稀疏高分辨率候选查询、ClusDet 的候选区域后续裁切，以及 DMNet 的粗粒度密度/区域引导裁切思路；这里只借鉴“先粗定位、再把有限计算放到候选区域”的方法，不复现它们的检测头、数据集或精度结论：

- [QueryDet CVPR 2022 论文](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html) / [官方代码](https://github.com/Small-Object-Detection/QueryDet)
- [ClusDet ICCV 2019 论文](https://openaccess.thecvf.com/content_ICCV_2019/html/Yang_Clustered_Object_Detection_in_Aerial_Images_ICCV_2019_paper.html) / [官方代码](https://github.com/fyangneil/Clustered-Object-Detection-in-Aerial-Image)
- [DMNet 官方代码](https://github.com/Cli98/DMNet)

本任务只有 RGB，不能把颜色阈值当作果实检测器。热图只是候选区域排序；树叶边缘也可能产生响应，因此 E08 保留全图、保留一个覆盖未覆盖区域的窗口，并在平坦或非有限热图时回退到固定窗口。

## 实现

`citrus_crop_guide.py` 中的 `TinyCropGuide` 只有四个普通卷积块和一个单通道预测层（约 15.4K 参数，384 输入约 0.106 GFLOPs）。训练目标是由训练集实例框中心生成的 stride-8 高斯热图，使用前景/背景质量平衡 BCE；训练固定 20 epoch、AdamW、AMP=False，选择最后一轮，不读取验证/测试标签。引导器输出只用于窗口排序，分割器仍采用 E03 的网络和相同的固定超参数。

每张原图的视图预算固定为 5 次：

1. 全图；
2. 热图最高响应的 3 个窗口，窗口之间做响应衰减以减少重复；
3. 一个覆盖尚未覆盖区域的窗口。

窗口大小仍为原图两个轴的 0.6，分割训练中局部视图概率为 0.5，因而 E08 与 E03 只改变窗口位置，不改变网络 YAML、优化器、AMP、cache、epoch 长度或切片预算。派生标签仍使用 Shapely 相交；无效/不连通/含孔的局部多边形整张派生视图拒绝并保留全图，不进行额外最小像素或面积清洗。

## 当前配对诊断

在同一份 SAGE80 权重和同一验证集上进行的配对诊断（不是 E08 的 300 epoch 训练结果）中：

| 输入方式 | Mask AP50 | 严格 Mask AP | tiny R@0.25 |
|---|---:|---:|---:|
| 全图一次 | 81.404% | 59.459% | 17.391% |
| 固定四切片 | 74.833% | 60.070% | 40.994% |
| RGB 热图引导切片 | 75.892% | 61.247% | 41.615% |

相对固定切片，热图引导提高了 1.059 个百分点的 AP50、1.177 个百分点的严格 AP，tiny 召回多匹配 1 个实例；但仍低于一次全图的 AP50，且引导方案仍需要 5 次分割前向。因此它目前是一个有证据的对照方向，不是已经证明的最终部署方案。

## 运行与比较

批量入口是 `20260907_citrus_e_batch.py`，默认固定协议为 `AMP=False`、`cache=True`、300 epoch、seed42，支持 VS Code 前台三角形运行。先运行 `SUITE="guided"`，它只比较 `E03_phase_sliced` 与 `E08_phase_guided_sliced`；确认无误后再运行 `SUITE="all"` 完成 E00–E08。首次运行会在项目目录生成 `_crop_guide/guide.pt` 和 `guide.json`，随后准备 `_prepared_guided_views`。引导器训练日志应明确显示 train-only cache、20 个 epoch 和 `amp=False`。

正式结论应同时报告整图 AP、配对切片 AP、tiny 召回、重复/误检率和实际延迟。若 E08 训练后只提升 tiny 召回而 AP50 不升，应优先检查热图候选偏差和切片边缘实例，而不是继续堆叠注意力模块。

## 单卡绑定

批量入口把 `DEVICE="1"` 作为物理 GPU 1 固定到 `CUDA_VISIBLE_DEVICES=1`，随后 Ultralytics 在这个单卡可见环境中使用逻辑 `cuda:0`。不要再把入口改成 `DEVICE="0"` 来“映射”物理卡，也不要在同一终端预先设置 `CUDA_VISIBLE_DEVICES=0,1`。如果旧进程已经启动，必须先用 `Ctrl+C` 停止并在新终端重新启动；CUDA 可见设备不能在已初始化的进程中动态切换。运行日志应同时记录物理请求卡号和 `loaded_data_summary.json` 中的实际 GPU 名称。
