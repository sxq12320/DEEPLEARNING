# T 系列：跨系列统一复跑集合

T 系列用于把历史高分模型、清洗数据上的代表模型和当前结构候选放在相同训练协议下重新训练。
所有 YAML 都是对应来源模型的独立快照，并显式锁定 `scale: n`。结果默认写入
`1_results/T_series/`，不会覆盖原系列结果。

## 入选模型

| T编号 | 来源 | 选择原因 | 当前角色 |
|---|---|---|---|
| T00 | 官方 YOLO11n-seg | 必须存在的统一基线 | 数据与协议锚点 |
| T01 | F14 | 历史最强单因素 SPPF-LSKA | 复核 LSKA 的真实收益 |
| T02 | G10 | 历史数据总体最强组合 | 最关键历史复现 |
| T03 | N02 | N 系列历史最优 | 第二种历史高分结构 |
| T04 | L06 | LSKA、上下文和拓扑组合 | L 系列代表 |
| T05 | S04 | 清洗数据上稳定且轻量 | 简单轻量 head 对照 |
| T06 | B06 | 清洗数据 B 系列最优 | 当前实证较优锚点 |
| T07 | C03 | 语义/细节双原型任务核心 | 遮挡凹掩膜与 split/merge 候选 |
| T08 | D06 | 形状—语义主干精度候选 | 主干重构候选 |
| T09 | D07 | D06 主干的轻量版本 | 部署候选 |

旧 F/G/N 与清洗数据 S/B 的历史数值来自不同数据或协议，只用于解释入选原因，不能直接排名。

## 数据路径

数据集路径完全由运行者通过 `--data` 指定。脚本没有默认数据集路径，也不要求数据指纹或固定图片数量。
它只解析指定的 `data.yaml`，检查 train/val/test 路径是否能够读取，然后使用同一路径开始训练。

## 服务器运行

进入服务器上的代码目录：

```bash
cd /你的路径/1_SEVER/code/ultralytics-main-new
pip install -e .
```

先让十个模型各跑 3 轮 smoke：

```bash
python -u 20260829_citrus_t_batch.py \
  --data /你自己的数据集路径/data.yaml \
  --suite smoke --epochs 3 \
  --batch 16 --workers 4 --device 0 \
  --project 1_results/T_series/CITRUS_T_SMOKE_3EP
```

smoke 全部正常后，批量训练 300 轮：

```bash
nohup python -u 20260829_citrus_t_batch.py \
  --data /你自己的数据集路径/data.yaml \
  --suite all --epochs 300 \
  --batch 16 --workers 4 --device 0 \
  --project 1_results/T_series/CITRUS_T_ALL_300EP \
  > t_all_300ep.log 2>&1 &

tail -f t_all_300ep.log
```

脚本按 T00 到 T09 严格串行，同一张 GPU 不会同时启动两个模型。中断后使用同一条命令重启：
已经完整结束的模型会跳过，存在 `last.pt` 的当前模型会续训。

## 锁定训练协议

所有模型使用相同的 `data.yaml`、`yolo11n-seg.pt` 初始化、AdamW、`lr0=0.001`、
`weight_decay=0.0005`、`imgsz=640`、`dropout=0`、`amp=False` 和 seed 42。
模型自带的明确配套损失保持各自设置。
