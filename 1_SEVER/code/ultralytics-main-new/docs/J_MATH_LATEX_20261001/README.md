# J11 数学推导与图解

本文件夹是研究技术说明，不是正式投稿稿，也不是已验证最强模型的声明。

## 先看哪个文件

- `J11_math_derivation.pdf`：完整中文推导，包含命题、证明、反例、模块图与证据边界。
- `J11_math_derivation.tex`：标准可编辑 LaTeX 源码。
- `figures/`：矢量 PDF、可编辑 SVG、600 dpi PNG；CSV 为定量图源数据。
- `verification.json`：源文件 SHA-256 和 SymPy 精确检查。
- `FIGURE_CONTRACT.md`：每图的科学问题、证据角色、禁止越界的结论。
- `qa/`：逐页预览、文本提取、最终检查；不作为论文主图。

## 编译

需要 XeLaTeX。主机已有 MiKTeX；本次没有安装新的 TeX 环境，也没有使用 ReportLab 伪装成 LaTeX。
在本文件夹运行两遍：

```powershell
xelatex -interaction=nonstopmode -halt-on-error J11_math_derivation.tex
xelatex -interaction=nonstopmode -halt-on-error J11_math_derivation.tex
```

使用 Windows 中文字体与 Times New Roman。其他系统可将 `fontset=windows` 替换为可用的 ctex 字体集。
迁移时同时复制 `figures/`，不能只复制 `.tex`。

本次内置 LaTeX 编译工具不可调用；打开 Codex 原生编辑器的工具也返回不可用。
已保存真正的 `.tex` 源码，并使用本机已安装的 XeLaTeX 成功编译；未声称编辑器已经打开。

## 科学边界

1. J11 是综合候选，不是精度冠军：只有参数/FLOPs/CPU 构建诊断，没有正式三 seed 精度或 GPU 延迟。
2. 灰度/对比是 RGB 的确定性表示，不增加新的传感信息。
3. 重排本身可逆，随后学习压缩不一定可逆。
4. 凸混合界不证明完整网络稳定；差分消常量不保证消叶片。
5. 高分辨率原型不能自动制造检测候选；P1/P2 当前不是检测头。
6. tiny/边界辅助损失为可选 J14–J16，不冒充 J11 已使用的项。
7. 梯度预算只是一项明确有条件、未实现的未来建议，不是 AdamW/全局收敛证明。
8. 历史切片诊断包含推理合并变化，来自不同旧验证成员；不与正式 grouped_dedup 或 J11 直接排名。
9. PR 横轴是 Recall；保留真实评估端点，不篡改 PR/AP。

文档制作没有修改模型、训练脚本、数据集或已完成结果。
