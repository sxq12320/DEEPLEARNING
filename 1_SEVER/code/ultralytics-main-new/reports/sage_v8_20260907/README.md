# V8证据与验证索引

先读 `RESULTS.md` 和主代码目录 `docs/SAGE_V8_DESIGN.md`。

- `audit.json`：V8改动前的174次运行、271个YAML及V7R源文件一致性快照。
- `history.csv` / `history_source_mapping.json`：历史结果表及原配置文件名映射；不同协议不能混排。
- `diagnostics/` / `size_conditioned_colour.json`：70/72/76的逐实例尺度、颜色、接触/凹陷代理诊断。
- `pr70/diagnostic.json`、`pr76/diagnostic.json`：实测PR点与原始框/评分/NMS/掩膜失败分解。
- `pr70/empirical_mask_pr.svg`、`pr76/empirical_mask_pr.svg`：不添加虚拟零尾巴的诊断图，原AP不变。
- `benchmark_cpu_final.json`：有效的独立CPU交错性能测试；不要引用初次benchmark_cpu.json。
- `regression.log` / `tests_v8_final.log` / `tests_batch.log`：兼容性、最终V8、队列与真实回调测试。
- `smoke_final/` / `smoke_final.log`：四个新模型各一轮4图真实train/val/save；不是正式精度实验。
- `dry_run.log`：批量入口五个模型构建/GFLOPs验证。
- `initialization/` / `delivery_reference_sha256.json`：实际初始化参数相等率、交付源码及阅读的本地参考源码哈希。

保留早期 `smoke_new/`、`benchmark_cpu.json` 作为调试记录。它们早于最终stem增益登记和输入padding修复，
不能作为最终模型、权重或性能凭据。V8正式训练尚未执行，请使用根目录RUN_SAGE_V8.py新建结果目录。
