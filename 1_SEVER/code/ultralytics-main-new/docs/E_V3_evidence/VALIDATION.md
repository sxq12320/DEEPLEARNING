# E V3 工程验证记录

验证时间：2026-09-09（本地 Python 3.9.13，Torch 2.8.0+cpu）。

| 检查 | 结果 | 说明 |
|---|---|---|
| 8 个 YAML 构建 | 通过 | `--suite all --dry-run --device cpu`；参数 2.170–2.332M，9.861–10.130 GFLOPs@640 |
| forward/loss/backward | 通过 | `pytest tests/test_citrus_e_v3.py -q`，29 passed |
| fused forward 对齐 | 通过 | 每个配置比较 fuse 前后框、分数、mask 系数，容差 rtol=.002、atol=.02 |
| 预训练加载/保存 | 通过 | 8 个 YAML 都检查关键共享层和 checkpoint round-trip |
| Q 评分逻辑 | 通过 | 预测框裁剪目标、质量开关、消失实例 ID 不重编号 |
| 1 epoch 真实链路 | 通过 | 从实际数据复制 4 train/2 val 图像到临时目录；8 个模型均有 completed.json、results.csv 和 paired_sliced_eval |
| 静态检查 | 通过 | Ruff E9/F/I 对本次新增和改动 Python 文件无错误 |

短训练临时产物在 `E:/mastercode/_work/citrus_ev3_smoke_20260909_r2`，没有写入正式 `results` 或数据集。短训练指标为零是因为只用 2 张验证图且只训练 1 轮，不能用于精度结论。

本地 CPU 的完整基准在 `cpu_profile.json`：E30 为 2,323,380 参数/10.097 GFLOPs，E37 为 2,179,145 参数/9.893 GFLOPs。训练前向+loss+反向中位数约 749 ms 对 697 ms；融合前向约 183 ms 对 184 ms。因此没有声称 CPU 或服务器 GPU 一定加速，正式机器需按相同 batch、尺寸、精度和数据加载条件重新测量。

服务器正式实验仍需由用户运行 300 epoch；本地验证没有、也不可能替代真实 GPU 结果。建议先把 `RUN_CITRUS_E_V3.py` 的 `SUITE` 设为 `"smoke"` 做一次服务器环境检查，再运行 `"priority"` 筛选单因素，最后才跑 `"all"`。
