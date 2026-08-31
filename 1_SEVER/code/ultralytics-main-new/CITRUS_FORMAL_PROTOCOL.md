# 柑橘论文一正式训练协议 v1

机器可读源文件：`protocols/citrus_paper1_formal_v1.yaml`  
协议 ID：`citrus_paper1_rgb_groupaware_v1`

本协议适用于今后所有 RGB 未成熟柑橘实例分割结构实验。模型 YAML 只负责定义网络；不得在不同模型之间偷偷改变训练超参数。

## 固定训练超参数

| 类别 | 参数 | 固定值 |
|---|---|---|
| 输入 | `imgsz` | 640 |
| 批次 | `batch` / `nbs` | 16 / 64 |
| 数据加载 | `workers` / `cache` | 4 / false |
| 初始化 | checkpoint | 同一个 `yolo11n-seg.pt` |
| 优化器 | `optimizer` | AdamW |
| 学习率 | `lr0` / `lrf` | 0.001 / 0.01 |
| 优化器参数 | `momentum` / `weight_decay` | 0.937 / 0.0005 |
| warm-up | epochs / momentum / bias LR | 3.0 / 0.8 / 0.1 |
| 基础损失 | box / cls / dfl | 7.5 / 0.5 / 1.5 |
| 数值精度 | `amp` | **false** |
| 正则 | `dropout` | 0.0 |
| 可重复性 | deterministic | true |
| early stop | `patience` | 300，保证300轮正式实验不会因模型不同提前停止 |
| mask | overlap / ratio | true / 4 |
| 调度 | cosine / multi-scale | false / 0.0 |
| mosaic | probability / close | 1.0 / 最后10轮关闭 |

## 固定增强

| 参数 | 值 | 参数 | 值 |
|---|---:|---|---:|
| hsv_h | 0.015 | hsv_s | 0.7 |
| hsv_v | 0.4 | translate | 0.1 |
| scale | 0.5 | fliplr | 0.5 |
| degrees | 0.0 | flipud | 0.0 |
| shear | 0.0 | perspective | 0.0 |
| bgr | 0.0 | mixup | 0.0 |
| cutmix | 0.0 | copy_paste | 0.0 |

## 固定实验阶段

| 阶段 | Epoch | Seed | 用途 |
|---|---:|---|---|
| 构建/反向测试 | 0 | 42 | YAML、前向、反向、GFLOPs |
| smoke | 1--3 | 42 | 排除路径、NaN、显存问题 |
| screening | 50 | 42 | 结构筛选，只运行一次 |
| final | 300 | 42、43、44 | 基线与最终方法，报告 mean±std |

## 允许改变的内容

- 模型 YAML；
- 输出目录和实验名称；
- 服务器 `device`；
- 数据集在不同机器上的绝对路径，但 split 成员必须完全相同；
- 预先声明的 epoch 阶段和 seed；
- 单独编号、具有配对对照的 loss/assignment 实验。

## 禁止混入结构比较的变化

AMP、batch、imgsz、预训练权重、优化器、学习率、dropout、增强、mask_ratio 和 patience 均不得按模型调整。若显存不足，不能只降低某个大模型的 batch 后继续放在同一张表中；必须让整组使用新的协议版本重跑。

方法专属辅助损失也必须完整记录。为了避免“命令行数值不同”，同一系列可把所有候选 loss gain 固定传给全部模型；模型没有对应输出时该项自然为零。若要判断收益来自结构还是 loss，必须另外建立 `loss-off` 配对消融。

## AMP 规则

正式协议固定 `amp=false`。AMP 通常用于速度和显存，不应被当作网络创新。AMP on/off 只能在同模型、同 split、同 seed、同 batch 和同超参数下成对运行，并单独标记为数值协议实验，不能与正式结构结果混合。

每次训练会在结果目录 `_protocol/` 保存 `formal_protocol.yaml` 和 `formal_protocol.sha256`。论文表格中的结果如果协议签名不同，默认不可直接比较。
