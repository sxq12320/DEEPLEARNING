# PROGRESS —— 机械原理 Manim 课程（2_mechanical）

状态：**14 讲全部场景代码完成；每讲至少 1 个代表性场景低画质渲染验证通过。**
未做：全部 ~97 场景的逐条渲染与高画质成品输出（按需再跑）。

## 已完成

- [x] OUTLINE.md —— 14 讲大纲（覆盖孙桓八版全书 14 章，页码锚点）
- [x] mechlib v2 —— style/primitives/solvers/curves/mechanisms/formulas 六模块
- [x] L01–L14 共 ~97 个场景类（docstring 即讲稿+页码）
- [x] XeLaTeX+ctex 中英混排公式管线（`mtex()`）
- [x] 冒烟测试 + 每讲抽样低画质渲染

## 抽样渲染记录（480p15 验证通过）

| 讲 | 已渲染场景 | 关键验证点 |
|----|-----------|-----------|
| _smoke | S01_Smoke | 中文字体/公式/机构/齿轮全链路 |
| L01 | S02_EngineDissect | 内燃机四机构拆解 |
| L02 | S06_VirtualConstraint | 虚约束标注位置 |
| L03 | S03_Kennedy | 三心定理反证动画 |
| L04 | S03_ScrewIncline | 螺旋展开+拧紧公式排版 |
| L05 | S01_Unbalance | 弹簧上转子不平衡 |
| L06 | S04_Flywheel | 飞轮+曲柄滑块联动 |
| L07 | S06_QuickReturn, S07_PressureAngle | 急回导杆、γ 实时标注 |
| L08 | S03_InversionMethod | 反转法作图 |
| L09 | S02_MotionLaws | s/v/a 四规律网格（已重排版式） |
| L10 | S02_InvoluteBirth | 绳线展开生成渐开线（方向已修） |
| L11 | S02_Generating | 范成包络（齿条刀方向/尺寸已修） |
| L12 | S03_InversionMethod, S05_Differential | 转化轮系、差速器 |
| L13 | S02_Geneva | 槽轮间歇+锁止弧 |
| L14 | S05_EngineReunion | 全机合体回收 |

## 本修过的缺陷

- `gear_profile` 根圆弧 arctan2 在 ±π 回绕 → 齿面横穿长弦（已修）
- `angle_mark` dir2<dir1 时扫 ~300° 大弧 → 改走劣弧（已修）
- `MathTex` 走默认 LaTeX 无法渲染中文 → 全部换 `mtex()`（xelatex+ctex）
- 顶部标注 `to_edge(UP, buff≤1.25)` 撞标题分隔线 → 全部 ≥1.7
- L10 渐开线生成线方向反了 → 已修
- L11 齿条刀齿尖方向/成品齿轮与坯尺寸不匹配 → 已修
- L03 `DashedLine.point_from_proportion` → 改线性插值
- L14 飞轮/齿轮标签出屏 → 已修
- L04 螺旋公式与"底边"标注重叠 → 锚点下移

## v2.1 3b1b 化升级（已渲染验证）

- 基类升级 `MovingCameraScene`，新增 `zoom_to/zoom_reset` 镜头语言；
  落地于 L10 S02（推镜看渐开线生成）、L13 S02（推镜看销入槽）、
  L14 S05（9s 运镜推近整机）。
- 右上角常驻「缓转齿轮 + 集数徽章」ambient 微动元素（自动推导，全场景生效）。
- `takeaway()` 结论自动圈注脉冲；`emphasize/focus/glow` 强调三件套入库。
- L01 片头：大标题收缩为常驻角标；L02 大算例答案圈注；
  L05 配重脉冲；L07 死点红叉圈注。
- 复渲验证：L01 S01 / L02 S07 / L05 S01 / L10 S02 / L13 S02 / L14 S05 全部通过。

## v2.2 全量 QA 迭代（全场景低画质渲染+抽帧目检）

- **渲染覆盖**：L01–L14 全部 ~97 场景均完成 480p15 渲染（此前仅抽样）。
- **修过的崩溃**：`stroke=`/`strokewidth` 非法 kwarg（L06×2、L09×2）、
  `VGroup(点,点)` 传坐标数组（L08 S03）、L02 S08/S09 漏渲。
- **版式修复**（抽帧目检发现，~20 处）：
  - 顶部标注/公式撞标题分隔线：L07 S03、L09 S02（网格整体下移）、
    L10 S01（齿廓弧压低）、L11 S04、L12 S04、L13 S05、L14 S01、L02 S04。
  - 文本出屏/超框：L09 S01 分类树、L11 S05/S07 卡片改两行、
    L13 S04、L14 S01/S06、全部讲末悬念行统一 size 25。
  - 标注互相叠印：L09 S03/S04 改单标题 Transform 依次替换、
    L08 S02 中垂线标签改左下图例、L06 S01 三阶段标签错位排开、
    L06 S02 公式区下移、L06 S03 "亏功"标签进红区内部、
    L02 S04 问句左移避让放大圆、L02 S08 公式右移、
    L03 S05 速度多边形标注外法线分散、L01 S07 底部三行归组、
    L05 S04 导轨演示下移、L07 S01 机构下移、L12 S02 齿链上移、
    L13 S03 齿轮上移、L14 S04 要点右移、L14 S05 曲柄滑块标签改挂飞轮下。
  - 普通 Text 里的 `v_A`/`μ_l`/`J_e` 等下划线字面量 → 全文清理（~50 处），
    `mtex()` 内 LaTeX 下标保留。
- **新增强调点**：L03 S03 瞬心 `focus`、L12 S03 转化轮系 `emphasize`、
  L04 S05 自锁条件 `focus`。
- 全项目 `compileall` 通过；修复场景均复渲+抽帧复检通过。

## 下一步（用户侧）

1. 按 RENDER.md 逐讲 `-a` 低画质过一遍，发现版式问题再迭代；
2. 满意后 `-qh --fps 60` 出成品；3. 按场景 docstring 配音。

## v3.0 Kurzgesagt 配色迁移 + 位置精修（多时刻抽帧复检）

### 配色
- 全局改为深空藏青底 `#0B1026` + 高饱和马卡龙色系（黄/薄荷/品红/亮青/珊瑚橙/紫），
  语义分工：ACCENT=强调黄、GOOD=薄荷、BAD=品红、NOTE=亮青、MUTED=冷灰蓝
- 所有 BLACK 遮罩（槽轮槽、不完全齿轮）→ BG；原生色死导入清理完毕
- 场景内 YELLOW/GREEN/BLUE_B 等原生色全部换成调色板常量

### 位置精修（本轮回放 218 帧后复修）
- **L01 S05**：右侧 "link" 标签出屏 → 缩小+右缘钳制
- **L02 S01**：三机构标签重排——F=0/F=2/F=1 错层分布，F=1 右移钳制防出屏
- **L02 S05**：焊死标记改为 always_redraw 跟随滚子（AnimatedCamRoller 新增 roller_center()）
- **L07 S03**：链式 Transform 修复（ineq2→ineq1），takeaway 缩短防撞页码
- **L08**：K 设计标注与图例位置调整
- **L09 S01**：凸轮上移；S04 删死代码
- **L10 S01**：啮合定律全重写——节圆真正相切于 P、公法线倾斜 20° 过 K、
  两齿廓弧在法线上相切（此前几何关系全错）；S03 啮合线过节点倾斜 20°；
  S04 齿轮缩小+四圆标签四象限分布；S06 啮合区间加矩形标注+引线
- **L11**：范成/变位三处布局微调
- **L12 S04**：公式左移+行星轮系缩小避让；S05 差速器弧形箭头 buff=0 修崩溃 + 收紧贴轮
- **L13 S01**：棘轮棘爪贴合验证
- **L14 S03**：串联链图验证干净

## v3.1 位置精修续轮（新配色下复检）

- **AnimatedGeneva 槽位几何 bug**（真缺陷）：`slot.rotate(ang, about_point=c2)`
  把槽条绕轮心二次摆动 → 4 条槽坍缩到 2 个角位、有槽条悬空轮外；
  改绕槽心自转 `slot.rotate(ang)` + 轮盘填充 0.45 + 槽长 0.72r@0.64r，
  现在 4 条径向槽正确均布、槽口恰到轮缘
- **L03 S05 速度多边形**：三角形放大（vA×0.75、vBA 1.2）、极点右下移
  (2.0,−1.55)、标注外法线偏移 0.62→0.95、size 22→20——三标注不再压边
- **L11 S02**：底部结论右移量 2.7→1.4，不再碰页码
- **L12 S05**：`Arrow(path_arc=…)` 默认 buff 触发 pointwise_become_partial
  广播错误崩溃 → buff=0 + 弧线收紧贴轮缘
- **L07 S03**：链式 Transform 修复（ineq2 未在屏上，应继续 Transform ineq1），
  takeaway 缩短防撞页码
- **L01 S05**："link" 标签右缘钳制防出屏；L02 S01：F=1 标签左移入画
- **配色清理**：mechlib 三个模块的 BLACK/YELLOW/ORANGE/BLUE_B 死导入移除，
  style.py docstring 示例色同步
- 全课程 102 个 mp4 已在新配色下重渲；compileall 通过

## v3.2 配色切换：Kurzgesagt 浅色版（米白纸质底）

- BG 深空藏青 #0B1026 → 米白 #F7F1E3；新增 INK #1E2A45 墨色
- 所有 WHITE 默认色（ctext/bullets/vec/pin_joint/fixed_pin/spring/motor/
  angle_mark/tag/vec_triangle…）→ INK；场景文件 9 处 WHITE → INK
- 高饱和色降明度适配浅底：黄→琥珀 #E8A000、薄荷→#00A381、品红→#E0306B、
  青→#0B9BD8、紫→#8B5CF6、珊瑚→#E0673F、机架灰→#5A6478
- takeaway 结论条文字改 INK（琥珀底字在米白上对比度不足），圈注环保持 ACCENT
- 槽轮槽口（BG 填充）在米白底上读作真正的开口槽，更直观
- L07 S01 连杆曲线椭圆与要点文本碰撞 → 四杆缩小下移修复
- 全课程 109 个 mp4 已在浅色主题下重渲；抽查 8 场景版式/可读性通过
