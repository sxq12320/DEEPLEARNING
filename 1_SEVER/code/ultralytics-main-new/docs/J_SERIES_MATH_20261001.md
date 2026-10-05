# J 系列数学依据：可证性质、任务假设与实验边界

日期：2026-10-01。对应 `ultralytics/nn/modules/citrus_j.py`、`ultralytics/utils/citrus_j_loss.py`。

这是一份可复核的**方法推导草稿**，不是涨点证明。卷积和 Transformer 在其他任务中有效，不推出它们的任意组合在柑橘任务中有效；论文必须分别给出机制、代码和同协议实验证据。

## 1. 从任务量化到设计约束

正式 grouped_dedup 数据共 965 图、5,890 个可见果实实例。640 letterbox 下：短边 <16 px 占 17.39%，solidity <0.85 占 17.61%，可见掩膜邻间隙 ≤2 px 占 30.95%。颜色低对比启发式子集占 11.46%，但它不是人工标注的“果叶混淆”类别。

因此需要同时满足三条约束：保留细节，避免叶枝背景被当成果实，不把相邻果实粘连或把遮挡造成的凹陷强行填圆。以下推导只针对这些约束，不引入深度、果梗、多任务姿态或不可见部分补全。

## 2. 灰度结构线索：归纳偏置，不是凭空增加信息

令 RGB 图像为随机变量 \(X\)，实例标注为 \(Y\)，附加结构线索 \(G=\phi(X)\)。由于 \(G\) 是 \(X\) 的确定性函数：

\[
H(Y\mid X,G)=H(Y\mid X),\quad I(Y;X,G)=I(Y;X).
\]

单独使用灰度满足数据处理不等式 \(I(Y;G)\le I(Y;X)\)。所以“RGB 加灰度获得额外信息”是错误表述。有限容量网络可以因显式结构表示而更容易拟合，但这属于**学习效率假设**，必须用 J06 对 J05 验证。

代码计算

\[
L=0.2989R+0.5870G+0.1140B,\quad C=L-\operatorname{AvgPool}_{5\times5}(L),\quad X_s=[R,G,B,L,C].
\]

这里 \(C\) 对空间恒定亮度偏移不敏感：若 \(L'=L+b\)，且池化对常数保持一致，则 \(C'=C\)。对乘性照明变化有 \(C'=aC\)，**不具有尺度不变性**；也不能保证区分同样具有纹理的叶片。RGB 原路保留，不使用预处理阈值删果实。

## 3. 相位重排：准确区分“可逆重排”和“有损压缩”

对偶数高宽输入，2×空间到通道重排定义为

\[
z_{4c+2a+b,i,j}=x_{c,2i+a,2j+b},\quad a,b\in\{0,1\}.
\]

向量化后 \(z=Px\)，其中 \(P\) 为置换矩阵，因而

\[
P^TP=I,\quad \|Px\|_2=\|x\|_2,\quad x=P^Tz.
\]

此结论只适用于 `pixel_unshuffle` 本身。后续通道投影为 \(u=Wz\)，若 \(W\in\mathbb{R}^{C_o\times4C_i}\) 且 \(C_o<4C_i\)，则

\[
\operatorname{rank}(W)\le C_o<4C_i,
\]

一般不可逆。BN/激活也不受前面的可逆性证明覆盖。因此不能把整个 JStem/JDown 称为“无损主干”，更不能宣称恢复了输入缩放已丢失的像素。

作用假设是：在压缩前显式呈现四个子像素相位，让网络学习怎样保留小果实证据。J03 对 J02 同时改变 stem 和下采样，验证的是**相位编码方案整体**，不能据此单独归因于 stem。

## 4. 非 CSP 主干：预算推导与速度边界

设宽度 \(C\)、空间面积 \(N=HW\)，仅对 \(\rho C\) 通道做 3×3 空间卷积，忽略取整、BN 和激活，其 MAC 为

\[
\operatorname{MAC}_{partial}=9\rho^2C^2N.
\]

JStage 用 \(\rho\approx1/4\)，但整个块还包含 value、gate 和输出三个 1×1 投影及一个深度卷积：

\[
\operatorname{MAC}_{block}\approx N[(9\rho^2+3)C^2+k^2C].
\]

这是块级估计，不是声称整网减少 16 倍。实际实现还有 BN、SiLU、concat、sigmoid、残差和内存访问。其形式为

\[
X'=X+\tanh(g)\,W_o[\phi(W_v\widetilde X)\odot\sigma(W_g\operatorname{DW}_k\widetilde X)].
\]

sigmoid gate 在 (0,1)，残差增益在 (-1,1)，但卷积权重及 BN 不受此限制，**不能推出网络输出幅值有界或训练收敛**。这借鉴 FasterNet 的部分空间混合与 GLU 的乘性选择，不是复现 FasterNet/SwiGLU，也不是新的普遍定理。

FLOPs 不等于延迟。当前 CPU FP32、小样本前向计时中 J11 没有证明比 YOLO11n 更快；最终必须用同一服务器 GPU、相同 batch、精度及后处理测量。

## 5. 一次竞争式融合：凸组合及梯度解释

将不同尺度特征投影并对齐至目标尺度：\(V_s=\mathcal R_s(W_sX_s)\)。代码对高分辨率来源使用池化，对低分辨率来源使用 nearest 上采样。源权重为

\[
\alpha_s(p)=\frac{e^{l_s(p)}}{\sum_r e^{l_r(p)}},\qquad Z(p)=\sum_s\alpha_s(p)V_s(p).
\]

因为 \(\alpha_s\ge0,\sum_s\alpha_s=1\)，逐位置、逐通道有

\[
\min_sV_s(p,c)\le Z(p,c)\le\max_sV_s(p,c),\quad
\|Z(p)\|_\infty\le\max_s\|V_s(p)\|_\infty.
\]

对固定 \(V_s\) 求源打分偏导：

\[
\frac{\partial Z}{\partial l_s}=\alpha_s(V_s-Z).
\]

它表示源权重的更新基于该来源相对混合结果的差异，**不表示权重自动对应果实/背景**。最终 DWConv/PWConv 之后的输出不再保证落在原凸包内。所有来源都计算，因此它不是稀疏 MoE，更不构成由 LLM MoE 推出的加速证明。

J08 对 J05 是相同参数图下竞争权重与均匀权重的对照。三个目标尺度直接从主干取证，不再串行做完整 FPN→PAN 往返；P3/P4/P5 都保留，P2 专供细节掩膜和场景支路。

## 6. 重要反例：单纯 key 去均值不会增强 softmax 对比

若一个窗口中的所有 key 都减去同一个均值 \(\bar k\)，则对固定 query

\[
q^T(k_j-\bar k)=q^Tk_j-q^T\bar k.
\]

第二项对全部 \(j\) 相同，故

\[
\operatorname{softmax}(q^T(K-\bar k)^T)=\operatorname{softmax}(q^TK^T).
\]

这是严格的 softmax 平移不变性。只有进一步改变 query、归一化、非线性、分组均值或注意力结构才可能产生不同效果。J 系列**没有采用这项无效的单独改动**。

## 7. 固定预算差分上下文：公共成分消除的条件性证明

在 stride16 特征上使用 \(m=4^2=16\) 个池化记忆位置，\(d=16\)。定义两个独立 query/key 流：

\[
A_+=\operatorname{softmax}(Q_+K_+^T/\sqrt d),\quad
A_-=\operatorname{softmax}(Q_-K_-^T/\sqrt d),\quad D=A_+-A_-.
\]

**两次 softmax 分别归一化**，所以 \(D\mathbf1=0\)。若 \(V=\mathbf1b^T+E\)，则

\[
DV=DE.
\]

空间恒定公共成分严格抵消。如果 \(\|E\|_\infty\le\epsilon\)，每行满足

\[
\|D_{i,:}\|_1\le2,\qquad\|(DE)_i\|_\infty\le2\epsilon.
\]

因此“近恒定成分”的影响有条件上界；树叶并不空间恒定，不能据此宣称消除任意树叶纹理或颜色混淆。真实果实信号也可能被抵消，这是设置 J09/J10 和 J11/J13 对照的原因。

代码残差 \(X'=X+0.25\tanh(g)W_o(DV)\)。给定学习后的 \(W_o\) 和有界值特征 \(M\)，单次校正满足

\[
\|X'-X\|_\infty\le0.5\|W_o\|_\infty M.
\]

这只是**单次幅度估计**。由于 query/key/值都随输入变化，且参数在学习，不能把它称为全网 Lipschitz 保证、PID 闭环稳定性或训练收敛定理。

注意力乘法复杂度从标准全局 \(O(N^2d)\) 降至 \(O(Nmd)\)，实际两次 QK 加一次 DV 的 MAC 约 \(3Nmd\)。投影仍需 \(O(NCd)\)。J 的结构只借鉴 Differential Transformer / LinearDiff 的差分与小记忆思路，**不复现原 VCA 的两阶段读写、位置 token、RMSNorm 和可学习 lambda**。原仓库第二阶段拼接后 softmax 与这里分别 softmax 也不同，不能把本节定理套到原作者实现。

## 8. 软场景分层：保留退路而非先阈值删除

浅层预测果实前景与可见边界 \(S\)，由现有实例标签合并得到监督，不新增叶片类别。识别特征为

\[
X_{cls}'=X_{cls}\odot[1+0.25\tanh(g_s)\operatorname{Pool}(\sigma(S_{fg}))].
\]

因为池化后概率在 [0,1]，乘性因子在 (0.75,1.25)，且零初始化增益使初始 \(X'_{cls}=X_{cls}\)。没有硬前景筛选，漏掉的软场景响应不会直接删除候选。框与掩膜系数的直接输入不被该门控修改，但共享主干和正样本分配仍有梯度耦合，**不是完全独立任务**。

这可以解释为借鉴“先组织场景证据，再识别与恢复细节”的计算顺序，不能当作人类视觉处理次序的已验证模型。J07/J05、J11/J12 检验是否有益。

## 9. 高分辨率掩膜与表示上限

每个实例掩膜 logit 使用 \(\ell_i(p)=\sum_{k=1}^{32}a_{ik}P_k(p)\)。J04 的原生 detail 解码为 stride4，J05 起结合 stride2 stem 输出 stride2 原型，而不是单纯放大最终二值掩膜。对一个 8×8 输入像素果实，网格支持从约 2×2 变成 4×4；不代表一定识别成功。

在同一图像、阈值与裁剪前，所有实例 logit 组成的矩阵满足 \(\operatorname{rank}(L)\le32\)。提升空间分辨率不解除原型数造成的表示限制，也不修复没有成功分配/输出候选的果实。这是把输入切片保留为公共训练协议而非宣称仅靠 decoder 解决全部漏检的原因。

## 10. 按实例归一化的小目标 Dice 与可见边界损失

令一个 GT 匹配 \(n_i\) 个正样本，先取系数平均 \(\bar a_i=n_i^{-1}\sum_r a_{ir}\)，再构成辅助掩膜。标准正样本 BCE 保留；**只有新增辅助项**按 GT 实例等贡献。

对 GT 框内像素，\(p_j=\sigma(\ell_j),y_j\in\{0,1\}\)，定义

\[
A=2\sum_jp_jy_j+\varepsilon,\quad B=\sum_jp_j+\sum_jy_j+\varepsilon,\quad
\mathcal L_D=1-A/B.
\]

实现 \(\varepsilon=1\)，其导数为

\[
\frac{\partial\mathcal L_D}{\partial p_j}=\frac{A-2y_jB}{B^2},\quad
\frac{\partial\mathcal L_D}{\partial\ell_j}=\frac{A-2y_jB}{B^2}p_j(1-p_j).
\]

可见 mask 面积折算回输入像素 \(a_i\)，权重

\[
w_i=\operatorname{clip}\!\left(\sqrt{256/\max(a_i,1)},1,2\right),\quad
\mathcal L_{tiny}=\frac{\sum_iw_i\mathcal L_{D,i}}{\sum_iw_i}.
\]

\(1\le w_i\le2\) 防止趋零面积产生无限权重；归一化权重比最多为 2。它不能保证梯度整体均衡或召回增长，且未匹配 GT 没有这一辅助损失。

边界带 \(B_i=\operatorname{MaxPool}(Y_i)-\operatorname{MinPool}(Y_i)\)，由**可见 GT**产生，对凹口与接触边界保留监督。边界 BCE 只在 GT 框内带状支持平均，不使用凸包或圆形补全。不能由这个局部损失直接推出 split/merge 错误下降，必须按难例子集测量。

总目标

\[
\mathcal L=\mathcal L_{standard}+\lambda_t\mathcal L_{tiny}+\lambda_b\mathcal L_{boundary}+\lambda_s\mathcal L_{scene}.
\]

J14/J15/J16 分别用 \((\lambda_t,\lambda_b)=(0.25,0),(0,0.1),(0.25,0.1)\)，场景启用时 \(\lambda_s=0.1\)。标准掩膜 gain 保持 7.5，新增项在它之后添加，其系数不被暗中再乘 7.5。

## 11. 优化器：采用成熟方法，不伪造控制理论创新

默认 AdamW 使用动量矩估计与解耦衰减，概括为

\[
\theta_{t+1}=(1-\eta_t\lambda)\theta_t-\eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
\]

J18 用 SGD 做单因子对照；J19 采用 warmup 后 cosine 调度。它们是训练机制对照，**不是新优化器**。不同优化器在同一 LR 下的筛选结果只说明该配方有效性，不能证明某优化器普遍优越。暂不加入无稳定依据的随机扰动/PID 学习率控制，以免把架构收益与训练调参混合。

## 12. 数学与实验闭环

`scripts/verify_citrus_j_math.py` 用 SymPy 1.14 验证融合导数、Dice 导数及 softmax 平移不变性，三个化简残差均为 0。`tests/test_citrus_j.py` 数值验证重排逆变换/范数、融合凸组合、常数值差分抵消与 key 去均值反例。

这些是可发表方法论的基础，但本身不是贡献充分性的证明。论文中的“有效、轻量、提高召回”必须由同划分、同初始化协议、同输入配方、同评价、三 seed 的实验支持；不能把旧泄漏验证的不同协议峰值放在一张涨点表中。

理论依据参见 [FasterNet](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Run_Dont_Walk_Chasing_Higher_FLOPS_for_Faster_Neural_Networks_CVPR_2023_paper.html)、[GLU Variants](https://arxiv.org/abs/2002.05202)、[Differential Transformer](https://arxiv.org/abs/2410.05258)、[LinearDiff](https://arxiv.org/abs/2511.00833)、[PIDNet](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html)。本文中的任务适配和条件性推导应与原方法明确区分。
