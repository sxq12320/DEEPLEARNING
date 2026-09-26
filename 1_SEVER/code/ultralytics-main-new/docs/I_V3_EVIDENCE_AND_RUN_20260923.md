# I V3：证据、设计与运行（2026-09-23）

## 1. 先区分事实与假设

盘点了 `1_SEVER/results` 中 299 个含 `results.csv` 的运行，原始清单见
`E:/mastercode/_work/20260923_i_v3/historical_results_inventory.csv`。
旧系列跨数据划分、训练轮数、AMP、初始化和评估代码，不可排一个总榜。特别是
I V2 五个 50 epoch 运行的 `val_loaded_files.txt` 与早期 `orange_yolo/val/images`
成员完全一致，而与正式 `orange_yolo_grouped_dedup_20260820/val/images` 不一致；
两者恰好都是 193 张，单看数量会误判。V3 必须重跑正式 grouped_dedup。

I V2 的 **50 epoch 最后一行**（旧划分、seed 42，数值为 Mask）：

| 模型 | AP50 | AP50-95 | Recall | 相对 I20 的 AP50-95 |
|---|---:|---:|---:|---:|
| I20 修正后三尺度对照 | 0.83340 | 0.69622 | 0.74071 | — |
| I21 关闭识别路由 | 0.83062 | 0.69604 | 0.75327 | -0.00018 |
| I22 原始 P2 候选 | 0.82761 | 0.69566 | 0.74833 | -0.00056 |
| I23 语义过滤 P2 | 0.83281 | 0.69580 | 0.75405 | -0.00042 |
| I24 大小核 P2 | 0.83244 | 0.69701 | 0.74452 | +0.00079 |

这些差别最多 0.079 个百分点，且只是单 seed/50 epoch，不能判定任何 P2
结构显著有效。I25–I29 尚未出现在这批完成结果中，不能算作失败或成功。
I V1 的 I00、E V12 的 V12_03 等在旧协议 300 epoch 的最后 AP50-95 约
0.719/0.717，但与 I V2 的 50 epoch 不能直接比较，更不能与正式新划分比较。
此前 I V2 的 TypeSafe 结构化审查（`E:/mastercode/_work/20260922_i_v2/typesafe_response.json`）
也将“先修协议正确性、再隔离 P2”排在首位，并不支持把 PR 绘图末端当成
模型 bug。该结果只是已有审查记录，不是 V3 的新远端评估或实验验证。

同一 I20 配对诊断在原图输入下，`<16×16` 像素（面积 `<256`）目标
仅检出 25/161，约 15.5%；细切片输入后约 84/161，但推理中位时间从约
138 ms 增至 417 ms，并引入更多背景误检。因此真正的瓶颈不仅是小目标像素
不足，还包括切片后果实与叶片的局部结构混淆和速度成本。切片有召回价值，
不是免费增益；V3 固定原有输入配方，只隔离特征提取/融合与损失因素。

PR 曲线横轴是 **Recall，不是 confidence**。当前 `metrics.py` 在 observed
maximum recall 之后追加 `(recall=1, precision=0)` 哨兵点用于 AP 积分；
I20 原图在 conf≈0.01 的实测最大 Mask Recall 为 0.9047，因此末端贴零
部分包含绘图约定，不能直接证明置信度 0.85 时模型精度突然清零。仍需正视
真实遗漏与误检，比较原始 PR 点、R@P≥0.90、极小目标 Recall 和 FP 类型。

## 2. 论文依据与保留边界

- [BDNet, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Guan_BDNetBio-Inspired_Dual-Backbone_Small_Object_Detection_Network_CVPR_2026_paper.html)
  使用颜色与边缘的功能分工、分层融合，适合启发绿色果实/叶片混淆与小目标；
  目前未找到可核实的官方源码。它是遥感**检测**，不能直接证明柑橘实例分割有效。
  V3 不复写 CAM/VCHM/OrSM/FFM，而测试由同一 RGB 增广图确定性生成的窄
  明度/局部对比/梯度支路是否提供互补信息。
- [HVPNet 官方代码](https://github.com/jiaweiXu1029/HVPNet) 研究显著/伪装
  目标的功能分支；其标准 SMT-t+MobileNetV2 和轻量双 MobileNetV2 都比
  本课题需要的附加支路更重，而且有 RGB-D/T 设置。仅借鉴职责分离，
  不挪用网络代码或声称同样的伪装检测结论。
- [YOLO26 官方说明](https://docs.ultralytics.com/models/yolo26/) 的小目标
  感知分配提示“候选匹配”值得单独验证，但 I38 仅测试代码中已有的 tiny
  assignment blend，不能冠以 YOLO26 的实现或效果。

控制论类比仅限“零起步、有界残差修正”：`IV3AsymGrayFuse` 的可学习增益
从零开始，并由 `tanh` 限幅，使初始 RGB 主干保留原函数；这**不是**
闭环控制器、稳定性证明或“PID 网络”。桌面控制原理 PDF 是扫描件，未把
未核实的书内公式写成依据。

## 3. 十臂可归因对照

| 组 | 唯一主要变化 | 用途 |
|---|---|---|
| I30 | I20 YAML 原样重放 | 新划分锚点 |
| I31 | 明度窄支路，只融 P2 | 双支路可行性 |
| I32 | 明度+局部对比，只融 P2 | 局部相对差异 |
| I33 | I32 延伸到 P3 | 浅/中层分阶段融合 |
| I34 | 明度+Sobel 幅值，P2/P3 | 梯度先验 vs 局部对比 |
| I35 | I33 改直接残差融合 | 空间门控是否必要 |
| I36 | I33 + NWD box blend 0.1 | 极小框回归损失 |
| I37 | I33 + tiny Dice 0.40 | 极小 mask 监督 |
| I38 | I33 + tiny assignment mix 0.2 | 候选分配 |
| I39 | I33 关闭识别路由 | 识别路径独立贡献 |

所有支路用原 RGB 生成，不引入新图像文件、伪深度或额外标注；仍是 RGB
单模态实例分割。检测头保持三尺度，避免把 I V2 尚未证实的 P2 检测候选
混进主假设。优化器固定 AdamW，不一面改结构一面改优化器；等主结构稳定
后再做独立学习率/优化器实验。

## 4. 使用与验证

把完整 `code/ultralytics-main-new` 和要训练的数据上传服务器。在
`RUN_CITRUS_I_V3.py` 编辑 DATA、
DEVICE、SUITE；默认 `DRY_RUN=False`，点击 VS Code 运行箭头即开始批量训练。
如需先检查模型构建，可自行改成 `True`。训练在当前 Python 进程依次执行，
不是 nohup。先 `priority` 五组 50 epoch；只有 I33 或 I34 在正式划分上
同时改善 Mask AP50-95、AP50、R@P≥0.90、极小目标召回且速度可接受，
才运行 `losses`/`recognition` 对照和 300 epoch/3 seeds 复核。不同轮数/seed
使用新 PROJECT，不覆盖结果。

V3 按 `DATA` 指定的数据集直接运行，不核对 split manifest 或历史划分成员。
训练引擎仍会记录实际加载的文件，便于事后核查。正式比较时请自行确保
各组使用相同的清洗数据路径；仅凭验证集图像数量无法确认划分相同。
本地已完成 10 YAML 构建+前向+GFLOPs、4 个代表反向、官方权重映射检查；
没有在本机 CPU 上声称完成真实训练或得到涨点。

Baseline 的服务器日志显示现代隔离环境安装的是已撤回 `rfdetr==1.4.0`
而代码要求 `1.4.0.post0`，不是 I V3 模型故障。先在服务器运行
`REPAIR_CITRUS_MODERN_ENV.py`，再
`RUN_CITRUS_BASELINES_AMP.py --preflight-only`；新 baseline 入口默认在
新 PROJECT 中运行 AMP=1/0 两份，保持 Legacy78 原配方。未收到服务器
修复后的日志前，只能称“修复脚本与本地契约验证完成”，不能称服务器已修好。
