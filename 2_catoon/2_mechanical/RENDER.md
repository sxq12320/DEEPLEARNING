# 渲染手册 —— 机械原理课程（2_mechanical）

环境：Windows Anaconda Python 3.9 + Manim CE 0.19.0 + XeLaTeX（MiKTeX）+ Microsoft YaHei。
在 Git Bash 下 `python.exe` 可能解析到 WSL 占位符；若失效，用 PowerShell（`python`）或
Anaconda 绝对路径 `E:\AppInstallion\0_4_annaconda\python.exe`。

## 单场景预览（低画质，首选）

```powershell
cd E:\mastercode\2_catoon\2_mechanical\L07
python -m manim -ql --fps 15 scenes.py S07_PressureAngle
```

产物在 `L07\media\videos\scenes\480p15\<Scene>.mp4`。

注意：manim 渲染成功也会以退出码 1 收尾（版本升级提示噪音），
以输出 "File ready at ..." 为准。

## 整讲渲染

```powershell
cd E:\mastercode\2_catoon\2_mechanical\L07
python -m manim -ql --fps 15 scenes.py -a     # 低画质全部场景
python -m manim -qh --fps 60 scenes.py -a     # 成品 1080p60（确认后）
```

## 配色规范（Kurzgesagt 浅色版）

- \BG = #F7F1E3\ 米白纸质底；\INK = #1E2A45\ 墨色正文/主描边——
  **浅底上不要用 WHITE**（会隐形），一律用 INK；也不要用纯黑
- 强调色均为降明度版：ACCENT #E8A000 琥珀黄、GOOD #00A381 薄荷、
  BAD #E0306B 品红、NOTE #0B9BD8 深青、GEAR_2 #8B5CF6 紫、LINK_D #E0673F 珊瑚橙
- 遮罩/挖孔一律 fill_color=BG（槽轮槽、不完全齿轮）；要切回深底只需改 style.py 一处

## 抽帧检查

```powershell
ffmpeg -y -ss 8 -i media\videos\scenes\480p15\S07_PressureAngle.mp4 -frames:v 1 chk.png
```

## 场景清单速查

每讲 `scenes.py` 内按 `S0X_Name` 命名；docstring = 讲稿要点 + 页码锚点（孙桓八版）。
讲稿节奏：`header()` 片头 → 机构动画/公式分步 → `hold()` 留白 → `takeaway()` 结论条。
配音时按 docstring 与 `hold()` 停顿填词即可。

| 讲 | 主题 | 场景数 | 教材章节锚点 |
|----|------|--------|--------------|
| L01 | 绪论·构件·运动副·简图 | 6 | §1, p1-12 |
| L02 | 自由度·三大陷阱·杆组 | 8 | §2, p13-25 |
| L03 | 平面机构运动分析 | 8 | §3, p32-50 |
| L04 | 力分析·摩擦·效率·自锁 | 7 | §4-5, p60-84 |
| L05 | 机械的平衡 | 6 | §6, p85-96 |
| L06 | 运转·速度波动·飞轮 | 7 | §7, p98-120 |
| L07 | 连杆类型与特性 | 9 | §8-1~8-4, p124-146 |
| L08 | 四杆设计 | 7 | §8-5, p147-162 |
| L09 | 凸轮机构 | 7 | §9, p165-186 |
| L10 | 渐开线与标准齿轮 | 7 | §10-1~10-5, p197-208 |
| L11 | 范成·根切·变位·空间齿轮 | 7 | §10-6~10-10, p208-236 |
| L12 | 轮系·差速器 | 6 | §11, p237-262 |
| L13 | 间歇与其他机构 | 6 | §12, p264-285 |
| L14 | 方案设计+收官 | 6 | §13-14, p286-312 |

## 已修过的坑（勿回退）

- 中文公式一律 `mtex()`（xelatex+ctex 模板，mechlib.style），不要直接 `MathTex`。
- `to_edge(UP, buff=...)` 顶部标注需 `buff≥1.7`（标题分隔线在 y≈2.4）。
- `angle_mark(dir1, dir2)` 自动走劣弧；不要再手动交换方向角。
- `gear_profile` 根圆弧已做 ±π 回绕处理（curves.py），改几何前先跑冒烟。
- 相邻原始字符串写 `\quad`/`\"` 等命令尾部留空格，防粘连成 `\quadz`。
- `DashedLine` 是 VGroup，没有 `point_from_proportion`；取点用线性插值坐标。

## 3b1b 化增强（v2.1，mechlib.style）

- `LessonScene` 继承 `MovingCameraScene`：`self.zoom_to(目标, scale=0.5)` 推镜、
  `self.zoom_reset()` 复位；`takeaway()` 前若在放大态会自动复位。
- 每场景右上角自动出现「缓转齿轮 + `L07·S07` 集数徽章」（从目录/类名推导）。
- `takeaway()` 结论条自带 Circumscribe 圈注脉冲。
- 强调三件套：`self.emphasize(mob)` 脉冲指示 / `self.focus(mob)` 圈注一闪 /
  `glow(曲线)` 霓虹三层描边（对已生成曲线做 FadeIn 叠加最出片）。
- 用法示例：L10 S02（推镜+glow）、L13 S02（入槽推镜）、L14 S05（运镜推近）。
