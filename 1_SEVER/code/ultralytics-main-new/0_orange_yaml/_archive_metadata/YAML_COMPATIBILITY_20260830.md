# YAML 全量兼容性报告

审计日期：2026-08-30  
审计范围：`0_orange_yaml/**/*.yaml`  
总数：203

## 结论

| 检查 | 通过 | 失败 |
|---|---:|---:|
| `YOLO(yaml, task="segment")` 标准构建 | 203 | 0 |
| 64×64 eval真实前向、有限输出检查 | 203 | 0 |
| `YOLO(yaml).load("yolo11n-seg.pt")` 后前向 | 203 | 0 |
| `MODEL_INDEX.csv` 路径覆盖 | 203 | 0 |

因此，当前203个YAML均能通过标准Ultralytics入口直接实例化和运行。

## 机器可读记录

- `1_results/_compatibility/yaml_build_20260830.csv/json`
- `1_results/_compatibility/yaml_forward_20260830.csv/json`
- `1_results/_compatibility/yaml_official_load_forward_20260830.csv/json`

## Light官方权重迁移

Light01–04的颈部比官方YOLO11短，预测头从第23层移动到第15层。YAML已加入显式
`pretrained_layer_map: {15: 23}`，因此单独使用官方API也能迁移形状兼容的分割头：

| 模型 | 加载张量数 |
|---|---:|
| Light00 | 363/431 |
| Light01 | 423/481 |
| Light02 | 225/351 |
| Light03 | 105/309 |
| Light04 | 225/393 |

Light03使用不同的轻量预测头，兼容张量少于标准Segment是预期现象，不是加载失败。

## 警告解释

部分历史YAML构建时显示 `no model scale passed. Assuming scale='n'`。它们没有声明多尺度字典，解析器按nano处理；这不是错误，所有模型均完成了构建和前向。新YAML应显式写明 `scale: n` 和 `scales`，避免该提示。

## 内容重复但不删除

`A_baselines/current` 与 `A_baselines/legacy` 有12组SHA-256完全一致的文件。它们用于兼容旧训练脚本和结果路径。删除会破坏历史复现，因此保留，但统计模型数量或实验结果时不得把同一结构算两次。

