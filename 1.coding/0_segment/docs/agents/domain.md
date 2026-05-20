# 领域文档

工程类技能在浏览代码库时应如何读取领域文档。

## 探索前，先读这些

- **`CONTEXT.md`** 位于 `1.coding/0_segment/` — 如果存在。
- **`docs/adr/`** 位于 `1.coding/0_segment/docs/adr/` — 读取与当前工作相关的 ADR。

如果这些文件不存在，**静默跳过**。不要提示缺失，`.grill-with-docs` 技能会在需要时延迟创建它们。

## 文件结构

单上下文项目：

```
1.coding/0_segment/
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-xxx.md
│   └── 0002-xxx.md
├── models/
├── datasets/
└── train.py
```

## 使用术语表中的词汇

输出涉及领域概念时（issue 标题、重构提案、测试名称），使用 `CONTEXT.md` 中定义的术语。不要使用术语表明确避开的同义词。

如果需要的概念尚未在术语表中，要么是你生造了项目不用的语言（重新考虑），要么确实存在缺口（在 `.grill-with-docs` 中标注）。

## 标记 ADR 冲突

如果输出内容与现有 ADR 矛盾，明确标记：

> _与 ADR-0007（xxx）存在矛盾——但值得重新讨论，因为…_
