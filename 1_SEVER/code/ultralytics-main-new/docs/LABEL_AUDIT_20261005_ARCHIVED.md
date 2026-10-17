# 标签审计资料已移出服务器上传目录

2026-10-11 为减小 `1_SEVER/code` 上传体积，原 `docs/LABEL_AUDIT_20261005/` 已完整归档到：

`E:/mastercode/1_SEVER/archives/code_cleanup_20261011/LABEL_AUDIT_20261005/`

打开 [标签审计页面](E:/mastercode/1_SEVER/archives/code_cleanup_20261011/LABEL_AUDIT_20261005/标签审计.html)。

HTML、两份 CSV、审计图片及逐图推理 JSON 均保留，共 4,991 文件；移动前后文件数、总字节数一致，
HTML 与两份 CSV 的 SHA256 已核对。资料不是训练依赖，不再随 code 上传服务器。
审计生成脚本仍在 `scripts/`，需要继续审计时将其 `--audit` 指向上面的实际归档路径。
这里的候选漏标结果不等于人工已确认标签，不据此自动改动数据。
