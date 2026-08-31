# 文献检索日志

检索日期：2026-08-30。用途：面向第一篇 RGB 柑橘幼果实例分割论文的架构决策，不用于生成虚构性能数字。

## 数据库记录

| 数据库 | 主题查询 | 下载 | 文件 | 备注 |
|---|---|---:|---|---|
| OpenAlex | small object instance segmentation | 50 | `openalex_small_object_instance_seg.json` | API 总命中 16,976，下载 relevance 前 50 |
| OpenAlex | camouflage instance segmentation | 50 | `openalex_camouflage_instance_seg.json` | API 总命中 1,185 |
| OpenAlex | citrus instance segmentation | 50 | `openalex_citrus_instance_seg.json` | API 总命中 483 |
| arXiv API | small object AND instance segmentation/object detection | 50 | `arxiv_small_object.xml` | relevance 排序 |
| Crossref | title query: small object instance segmentation, 2019+ | 50 | `crossref_small_object.json` | 检索很宽，只用于 DOI/出版信息交叉检查 |
| Semantic Scholar | 上述三个主题 | 0 | `semanticscholar_*.json` | 匿名 429；本机 key 仍 Forbidden，不计为成功来源 |

三库成功记录共 250 条，标题标准化后 242 条精确唯一记录、8 条精确重复。仍存在同一工作的预印本/会议版语义重复。随后以 CVF、期刊官网和本地仓库 README 做引用追踪，形成 `selected_papers.csv`。

## 质量控制

- 机制和原始实验数字优先引用 CVF、期刊页面、DOI、arXiv 原文或官方仓库。
- 博客、二次模块合集与 AI 报告不作为性能证据。
- 他数据集的提升只写成“原论文报告”，不推断本数据集会得到同样提升。
- 开源代码是否存在与能否合法复制是两个问题；详见 `repository_registry.csv`。
- 2026 年论文/预印本按当前日期核验，仍可能在后续版本变化。

