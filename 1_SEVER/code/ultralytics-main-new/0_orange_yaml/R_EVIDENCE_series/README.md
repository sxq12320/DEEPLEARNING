# R_EVIDENCE

8组三因素候选，未正式训练。A非对称颈部，B局部对比入口，C语义校正。
运行 RUN_R_EVIDENCE.py，默认先跑3个现有模型参照。切RUN_MODE="factorial"启动8组。
统一正式分组全标签、640整图、原生损失；不切片，不加入辅助损失。
证据与固定超参见 ../../docs/EVIDENCE_RESET_20261112/README.md。
