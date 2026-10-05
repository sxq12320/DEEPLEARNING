# J_V2数学性质：条件、证明与反例边界

日期2026-10-03。公式对应 `citrus_j2_structure.py` 与修复 `citrus_j2_refined.py`，不是泛化性能证明。
本次补充复制子空间、插值交换、排序不变性与反例；不将已成立的代数性质替代效果验证。

## 1. 冻结BN的结构重参数化

对输出通道c，令卷积 \(z_c=W_c*x+b_c\)，冻结BN参数为 \(\gamma_c,\beta_c,\mu_c,v_c\)。

\[
\operatorname{BN}(z_c)=\frac{\gamma_c}{\sqrt{v_c+\epsilon}}(z_c-\mu_c)+\beta_c
=\widetilde W_c*x+\widetilde b_c,
\]
\[
\widetilde W_c=\frac{\gamma_cW_c}{\sqrt{v_c+\epsilon}},\qquad
\widetilde b_c=\beta_c+\frac{\gamma_c(b_c-\mu_c)}{\sqrt{v_c+\epsilon}}.
\]

J2RepMix是3×3 depthwise、1×1 depthwise与identity-BN相加，内部无非线性。
将1×1权重零填充至3×3，identity为中心单位脉冲 \(\Delta\)：

\[
W_{\mathrm{deploy}}=\widetilde W_{3}+\operatorname{pad}(\widetilde W_1)
+\operatorname{diag}\left(\gamma_{\mathrm{id}}/\sqrt{v_{\mathrm{id}}+\epsilon}\right)\Delta,
\]
\[
b_{\mathrm{deploy}}=\widetilde b_3+\widetilde b_1+
\beta_{\mathrm{id}}-\gamma_{\mathrm{id}}\mu_{\mathrm{id}}/\sqrt{v_{\mathrm{id}}+\epsilon}.
\]

由卷积线性性得到eval模式精确等价（浮点舍入外）。训练BN依赖批次，不能如此推导；分支若含激活亦不能直接相加。
这是RepViT/RepNeXt已有思想的独立实现，不是本文首次证明。

## 2. 中心—周围残差，不等于全网络消色不变性

局部平均算子A在边缘按有效邻域归一，因此 \(A\mathbf1=\mathbf1\)。定义 \(D=I-A\)：

\[
D(x+c\mathbf1)=Dx,\qquad
y=x+\tfrac14\tanh(a)\,Q(Dx).
\]

常量偏置在D输出中抵消；网络仍保留原始x，不会抹去RGB线索。
\(a=0\) 时模块严格恒等，投影参数的第一步梯度为零；gain可能得到梯度，但信号与上游梯度正交时也为零。
训练BN仍可能更新运行统计；不能称所有分支状态冻结或保证下一步必学习。
实际叶枝并非常量、纹理和轮廓与果实重叠，所以该恒等式**不保证背景被消除**。
Q含BN/学习投影，原始RGB前面还有非线性stem，因此不能外推为整网对光照/颜色绝对不变。

## 3. 细节优先有界融合

\[
u=P_d(d)+G\,\operatorname{Up}(P_s(s)),\quad
G=\tfrac12\operatorname{diag}(\tanh a_c),\quad y=R(u).
\]

对任意逐点向量p范数，\(\|G\|\le1/2\)，故

\[
\|u-P_d(d)\|_p\le\tfrac12\|\operatorname{Up}(P_s(s))\|_p.
\]

界只对u有效，比较对象已投影；P_s可放大任意倍，R也没有Lipschitz常数≤1的保证。
允许的G=.4、投影后s=−2.5d会令u=0，故有界gain并不证明细节不会被淹没。
初始化 \(a_c=\operatorname{atanh}(0.2)\) 得G=0.1；
\(\partial(\tfrac12\tanh a)/\partial a=1/(2\cosh^2a)\)。
非零G使语义分支可从第一步学习；不证明收敛、更不证明闭环控制稳定性。
本网络是单次前馈，不假称PID闭环控制系统。

## 4. 将卷积从高分辨率移回低分辨率

普通K×K卷积MAC约 \(HW C_i C_o K^2\)。空间尺寸各放大r倍后代价r²倍；
先通道投影再上采样，投影部分少r²倍MAC，r=2为4。
这不证明交换任意卷积和插值在数学上等价，它是显式改变结构的成本分析。
旧窄原型只在放大前做3×3，放大后用1×1投影；它形成下文的严格复制限制。
stride4只节省原型生成，标准loss/EV6仍插值到input/2；不得称实例mask匹配面积减少。

## 5. 共享头的参数与运算不是同一个概念

三尺度每个独立两层h通道3×3头的卷积权重为 \(3\cdot2\cdot9h^2\)。
共享后为 \(2\cdot9h^2\)，该部分权重降至1/3；尺度adapter、BN和输出层仍独立。
但各尺度仍需实际执行，MAC为 \(2\cdot9h^2\sum_lH_lW_l\)，不会因共享自动降至1/3。
J34的FLOPs降低还来自窄隐藏宽度、窄解码器和短颈部，不应归因给“共享”一词。

## 6. 为什么不能靠上采样补回观测证据

设I为原图，X为缩小输入，Y为目标状态，满足确定预处理链 \(Y\to I\to X\)，数据处理不等式给出
\(\mathcal I(Y;X)\le\mathcal I(Y;I)\)。
它说明压缩不能凭空增加信息；不说明所有图像都严格丢失任务信息，也不保证切片就会涨AP。
原图裁块重采样改变输入证据，但需承担视图数量、截断和误报；现有凝视对照只作为独立推理扩展。

边长d理想方形mask，在两轴各偏移δ时：
\[
\operatorname{IoU}=\frac{(d-\delta)^2}{2d^2-(d-\delta)^2},\qquad
\left.\frac{\partial\operatorname{IoU}}{\partial\delta}\right|_{\delta=0^+}=-4/d.
\]
条件是0≤δ<d、两轴同幅偏移；有符号位移用|δ|，原点为方向尖点。
相同绝对位移对更小方形更敏感。d=4/8/16、δ=1得9/23、49/79、225/287。
遮挡深凹mask不是理想方形；这只是动机解释，不能直接拟合其错误率。

## 7. 有界tiny权重，不重写分类分配

对匹配到的每个可见实例i，先平均其正anchor系数，再计算框内Dice \(\ell_i\)。
\[
w_i=\operatorname{clip}\big(\sqrt{256/\max(A_i,1)},1,2\big),\quad
L_{\rm tiny}=0.1\frac{\sum_iw_i\ell_i}{\sum_iw_i}.
\]
单实例权重上限2避免无界tiny重权；平滑Dice取值在[0,1]时该图附加项在[0,.1]。
但不保证梯度范数有统一上界，也不能监督未被分配到的实例。P2发现与此项不可互相替代。
平均系数的Dice不等于各推理候选Dice的均值；“等实例”只在每图内部，图间仍平均。
损失仍使用可见GT，无圆形先验、amodal填补、额外叶片标签。

## 8. 原型复制限制与相位修复的严格命题

设低栅格特征z∈R^(h×H×W)，nearest×2为N，逐点仿射为T(z)=Wz+b。
逐位置有T(Nz)=N(Tz)，因此旧decoder输出满足

\[
P_{c,2i+a,2j+b}=W_cz_{i,j}+b_c,\quad a,b\in\{0,1\}.
\]

任意空间无关系数C构成的CP继承这个等式；增加系数塔或改变loss不能打破结构约束。
这只约束未crop、未插值的logits，不是所有最终实例IoU的全局上界。
未crop logit还满足rank(CP)≤K，但二值化和实例crop不满足该简单秩结论。

修复将低栅格投影改为h→4K，PixelShuffle排列为channel=4c+2a+b：

\[
\widetilde P_{c,2i+a,2j+b}=W_{c,a,b}z_{i,j}+b_{c,a,b}.
\]

初始化令所有W_(c,a,b)=W_c、b_(c,a,b)=b_c，即得到旧函数的精确嵌入。
训练时参数不绑定。选择bias四相位[1,−1,−1,−1]就能表达同块一前景三背景。
因此新类包含旧类，且在局部logit表达上严格更大；它不保证优化能找到更优泛化解。

两个末层的卷积MAC都为4HW·hK。h=K=32时参数差为

\[
4K(h+1)-K(h+1)=3K(h+1)=3168.
\]

相同初始函数不等于同训练轨迹：旧共享W梯度为各相位梯度之和，新W_(a,b)分别更新。
额外Conv随机初始化立即被覆盖；实现用fork_rng保护CPU随机状态。ICNR只控制初始相位差异，
训练后仍可能出现棋盘伪影。checkpoint旧proto类路径不与新参数名自动一一匹配；本批次为scratch。

依据：[ESPCN原论文](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Shi_Real-Time_Single_Image_CVPR_2016_paper.html)、
[ICNR初始化研究](https://arxiv.org/abs/1707.02937)；这是已知算子的任务缺陷修复，不宣称首次发明。

## 9. 掩膜误差与IoU阈值：精确面积等式

设A=|G|、B=|M|、e=|G△M|，且A+B>0，则I=(A+B−e)/2，U=(A+B+e)/2，故

\[
\operatorname{IoU}=\frac{A+B-e}{A+B+e},\qquad
\operatorname{IoU}\ge t\iff e\le\frac{1-t}{1+t}(A+B),\quad 0\le t\le1.
\]

它严格解释了同样像素误差为何更易让小可见区域跌出匹配门槛，但不说明误差来自哪个网络环节。
对复杂遮挡区不应用“完整圆果面积”替代真实可见面积。

## 10. 系数装配与线性插值交换：等价及成本边界

设P∈R^(K×S)、C∈R^(M×K)、逐通道相同线性插值矩阵U∈R^(S×S')，则结合律给出

\[
C(PU)=(CP)U.
\]

因此保持后续sigmoid/crop/阈值/loss在同一栅格，两种实现的数学输出与梯度相同。
浮点归约顺序不同不保证bit-exact。先非线性再插值、提前crop或实例各用不同U都不受此结论保证。
先装配可把stride4→input/2的矩阵乘法面积减少4倍，但插值通道从K增为M，端到端可能并不加速。
本轮没有修改正式loss/EV6；单元测试以float64核验输出和两个输入的梯度。

当前标准重建每实例32·320²=3,276,800 MAC；M=300时约0.983G MAC。
共享塔减少参数而非逐尺度执行，P2/P3/P4共有33600位置，相对P3/P4/P5的8400为四倍。

## 11. 置信度校准、残差保真与噪声的反证

固定候选/几何/一对一匹配/NMS与保留集合，同一严格递增分数变换不改变TP/FP序列，所以AP不变。
固定conf阈值、ties或不同尺度变换改变保留集合时，前提不成立。不能靠统一抬高confidence恢复Rmax。

对f(x)=x+h(x)，只有另有Lip(h)=k<1才由三角不等式证明

\[
(1-k)\|x-y\|\le\|f(x)-f(y)\|\le(1+k)\|x-y\|.
\]

当前有界gain没有证明这个前提，不能据此声称双Lipschitz保真或控制稳定。
内部5×5均值A的某频率响应为−.25，D=I−A响应1.25；gain=.2、Q=identity对白噪声方差给出

\[
\operatorname{Var}(n+.2Dn)=\left[1+\frac{24}{25}(.4+.04)\right]\sigma^2=1.4224\sigma^2.
\]

这是允许参数的反例，不预测训练后Q；说明DC抵消不能证明叶脉误报降低。

## 12. 可复现核验

`python scripts/verify_citrus_j2_math.py --output <new_path>/math.json`，使用SymPy1.14精确表达式。
部署等价、常量偏置、zero-gain恒等及完整模型序列化另有PyTorch测试。
本轮新增 `tests/test_citrus_j2_refined.py`：复制约束、相位严格表达反例、初始化函数等价、
独立有限梯度、随机状态保护、线性插值交换及梯度等价。
独立数值反例脚本见 `_work/20261003_j2_deep/math_checks.py`；检查结果与研究精度严格分开。
不能从这些证明推出mAP提高10点；必须用固定协议、三seed、冻结test和难例子集实验来检验。
