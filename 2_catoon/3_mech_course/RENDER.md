# 渲染手册（在你自己方便的时候跑，本文档只给命令不自动执行）

> ⚠️ 渲染很吃 CPU/GPU，建议逐场景渲染、晚上挂机；先低质量预览再出成片。

## 环境
已确认可用：Manim CE 0.19.0 + MiKTeX（LaTeX 公式）+ Microsoft YaHei（中文）。

## 单场景预览（480p 快速看效果）
```powershell
cd E:\mastercode\2_catoon\3_mech_course\L01
manim -pql scenes.py S01_Opening
```

## 单场景成片（1080p60）
```powershell
manim -qh --fps 60 scenes.py S02_EngineDissect
```

## 整课批量（渲染该文件全部 Scene，逐个输出 mp4 供剪辑拼接）
```powershell
cd L01
manim -qh --fps 60 scenes.py -a
```
输出在 `media/videos/scenes/1080p60/`，按场景名各一个 mp4——你在剪辑软件里按 S01→S10 顺序拼接并配音。

## 建议工作流
1. `-pql` 预览某场景 → 感觉节奏不合适告诉我改 `self.hold()` 时长或拆分镜；
2. 满意后 `-qh -a` 整课出片（一节课约 30-90 分钟渲染，视机器）；
3. 配音时以每个 Scene 的 docstring（分镜讲稿要点+教材页码）为讲稿骨架。

## 课节清单
| 课 | 文件 | 场景数 | 主题 |
|---|---|---|---|
| L01 | L01/scenes.py | 10 | 绪论·机构组成·简图 |
| L02 | L02/scenes.py | 10 | 自由度·三大陷阱·杆组 |
| L03 | L03/scenes.py | 9 | 瞬心·图解·哥氏·解析 |
| L04 | L04/scenes.py | 9 | 四杆类型·Grashof·急回·死点 |
| L05 | L05/scenes.py | 7 | 四杆设计·连杆曲线 |
| L06 | L06/scenes.py | 8 | 凸轮规律·反转法·压力角 |
| L07 | L07/scenes.py | 9 | 渐开线·标准齿轮·重合度 |
| L08 | L08/scenes.py | 7 | 范成·根切·变位·斜齿蜗杆 |
| L09 | L09/scenes.py | 9 | 轮系·转化机构·差速器 |
| L10 | L10/scenes.py | 9 | 摩擦·效率·自锁 |
| L11 | L11/scenes.py | 9 | 平衡·飞轮·收官 |
| L12 | L12_bonus/scenes.py | 5 | 番外：间歇机构 |
