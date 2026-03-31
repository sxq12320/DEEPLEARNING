from manim import *
import numpy as np


# ============================================================
#  辅助函数：创建一层神经元
# ============================================================
def create_layer(num_neurons, color=BLUE, radius=0.25, spacing=0.85):
    """创建一列神经元（圆圈），返回 VGroup"""
    neurons = VGroup()
    for i in range(num_neurons):
        neuron = Circle(radius=radius, color=color, fill_opacity=0.3, stroke_width=2.5)
        neurons.add(neuron)
    neurons.arrange(DOWN, buff=spacing - 2 * radius)
    return neurons


def create_connections(layer_left, layer_right, color=GREY, stroke_width=1.2):
    """在两层之间创建全连接线"""
    lines = VGroup()
    for n_left in layer_left:
        for n_right in layer_right:
            line = Line(
                n_left.get_right(), n_right.get_left(),
                color=color, stroke_width=stroke_width, stroke_opacity=0.45
            )
            lines.add(line)
    return lines


def create_network(layer_sizes, colors=None, spacing=2.8):
    """
    创建完整网络结构
    layer_sizes: 每层神经元数 e.g. [3, 4, 4, 2]
    返回 (all_layers, all_connections, full_network_vgroup)
    """
    if colors is None:
        colors = [GREEN, BLUE, BLUE, RED]
    while len(colors) < len(layer_sizes):
        colors.append(BLUE)

    layers = []
    for i, size in enumerate(layer_sizes):
        layer = create_layer(size, color=colors[i])
        layers.append(layer)

    # 水平排列各层
    all_layers = VGroup(*layers)
    all_layers.arrange(RIGHT, buff=spacing)

    # 创建层间连接
    connections = []
    for i in range(len(layers) - 1):
        conn = create_connections(layers[i], layers[i + 1])
        connections.append(conn)
    all_connections = VGroup(*connections)

    full_network = VGroup(all_layers, all_connections)
    return layers, connections, full_network


# ============================================================
#  场景 1：全连接神经网络概述
# ============================================================
class FCN_Overview(Scene):
    def construct(self):
        # ---- 标题 ----
        title = Text("全连接神经网络", font_size=48, color=YELLOW)
        subtitle = Text("Fully Connected Neural Network (FCNN)", font_size=28, color=GREY_B)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3)
        title_group.to_edge(UP, buff=0.5)

        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(subtitle, shift=UP * 0.3), run_time=0.8)
        self.wait(0.5)

        # ---- 构建网络 [3, 5, 4, 2] ----
        layer_sizes = [3, 5, 4, 2]
        colors = [GREEN_C, BLUE_C, BLUE_C, RED_C]
        layers, connections, network = create_network(layer_sizes, colors, spacing=2.5)
        network.scale(0.75).next_to(title_group, DOWN, buff=0.8)

        # 层标签
        layer_names = ["输入层\nInput", "隐藏层 1\nHidden 1", "隐藏层 2\nHidden 2", "输出层\nOutput"]
        labels = VGroup()
        for i, name in enumerate(layer_names):
            label = Text(name, font_size=16, color=colors[i])
            label.next_to(layers[i], DOWN, buff=0.4)
            labels.add(label)

        # 动画：逐层显示
        for i in range(len(layers)):
            self.play(
                LaggedStart(*[GrowFromCenter(n) for n in layers[i]], lag_ratio=0.15),
                run_time=0.8
            )
            self.play(FadeIn(labels[i], shift=UP * 0.2), run_time=0.4)
            if i > 0:
                self.play(
                    LaggedStart(*[Create(c) for c in connections[i - 1]], lag_ratio=0.02),
                    run_time=0.6
                )

        self.wait(0.5)

        # ---- 核心特征说明 ----
        features = VGroup(
            Text("✦ 每一层的每个神经元与下一层所有神经元相连", font_size=20, color=WHITE),
            Text("✦ 每条连接都有一个可学习的权重 w", font_size=20, color=WHITE),
            Text("✦ 每个神经元有一个偏置 b", font_size=20, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        features.to_edge(DOWN, buff=0.5)

        for feat in features:
            self.play(FadeIn(feat, shift=RIGHT * 0.3), run_time=0.6)
            self.wait(0.3)

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1)


# ============================================================
#  场景 2：单个神经元 & 前向传播公式
# ============================================================
class FCN_SingleNeuron(Scene):
    def construct(self):
        # ---- 标题 ----
        title = Text("单个神经元的计算", font_size=42, color=YELLOW)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1)

        # ---- 画一个大神经元 ----
        big_neuron = Circle(radius=0.8, color=BLUE, fill_opacity=0.2, stroke_width=3)
        big_neuron.shift(RIGHT * 0.5)
        sigma_label = MathTex(r"\sigma", font_size=48, color=BLUE).move_to(big_neuron)

        # 输入箭头 x1, x2, x3
        inputs = VGroup()
        input_labels = []
        weights_labels = []
        num_inputs = 3
        start_y = 1.5
        for i in range(num_inputs):
            y = start_y - i * 1.5
            start = LEFT * 4 + UP * y
            end = big_neuron.get_left() + UP * (y * 0.3)
            arrow = Arrow(start, end, color=GREEN_C, stroke_width=2.5, buff=0.1)
            inputs.add(arrow)
            x_lab = MathTex(f"x_{i + 1}", font_size=32, color=GREEN_C).next_to(start, LEFT, buff=0.15)
            w_lab = MathTex(f"w_{i + 1}", font_size=26, color=ORANGE).next_to(arrow.get_center(), UP, buff=0.1)
            input_labels.append(x_lab)
            weights_labels.append(w_lab)

        # 偏置
        bias_arrow = Arrow(
            big_neuron.get_top() + UP * 1.0, big_neuron.get_top(),
            color=PURPLE, stroke_width=2.5, buff=0.1
        )
        bias_label = MathTex("b", font_size=32, color=PURPLE).next_to(bias_arrow, RIGHT, buff=0.15)

        # 输出箭头
        output_arrow = Arrow(big_neuron.get_right(), big_neuron.get_right() + RIGHT * 2.5,
                             color=RED_C, stroke_width=2.5, buff=0.1)
        output_label = MathTex("y", font_size=32, color=RED_C).next_to(output_arrow, RIGHT, buff=0.15)

        # 显示动画
        self.play(GrowFromCenter(big_neuron), Write(sigma_label), run_time=0.8)

        for i in range(num_inputs):
            self.play(
                GrowArrow(inputs[i]),
                FadeIn(input_labels[i]),
                FadeIn(weights_labels[i]),
                run_time=0.5
            )

        self.play(GrowArrow(bias_arrow), FadeIn(bias_label), run_time=0.5)
        self.play(GrowArrow(output_arrow), FadeIn(output_label), run_time=0.5)
        self.wait(0.5)

        # ---- 公式推导 ----
        formula_box = RoundedRectangle(
            corner_radius=0.2, width=10, height=3.5,
            color=YELLOW, fill_opacity=0.05, stroke_width=1.5
        ).to_edge(DOWN, buff=0.3)

        step1_title = Text("Step 1: 线性变换", font_size=24, color=YELLOW)
        step1_title.next_to(formula_box.get_top(), DOWN, buff=0.3).align_to(formula_box, LEFT).shift(RIGHT * 0.3)

        linear_eq = MathTex(
            r"z", r"=", r"w_1 x_1", r"+", r"w_2 x_2", r"+", r"w_3 x_3", r"+", r"b",
            font_size=34
        )
        linear_eq.set_color_by_tex("w", ORANGE)
        linear_eq.set_color_by_tex("x", GREEN_C)
        linear_eq.set_color_by_tex("b", PURPLE)
        linear_eq.set_color_by_tex("z", WHITE)
        linear_eq.next_to(step1_title, DOWN, buff=0.3)

        linear_vec = MathTex(
            r"z = \mathbf{w}^\top \mathbf{x} + b",
            font_size=36, color=WHITE
        )
        linear_vec.next_to(linear_eq, RIGHT, buff=0.8)

        step2_title = Text("Step 2: 激活函数", font_size=24, color=YELLOW)
        step2_title.next_to(linear_eq, DOWN, buff=0.4).align_to(step1_title, LEFT)

        activation_eq = MathTex(
            r"y = \sigma(z) = \sigma(\mathbf{w}^\top \mathbf{x} + b)",
            font_size=36
        )
        activation_eq.next_to(step2_title, DOWN, buff=0.3)

        self.play(FadeIn(formula_box), run_time=0.5)
        self.play(Write(step1_title), run_time=0.5)
        self.play(Write(linear_eq), run_time=1.2)
        self.play(Write(linear_vec), run_time=0.8)
        self.wait(0.5)
        self.play(Write(step2_title), run_time=0.5)
        self.play(Write(activation_eq), run_time=1.2)
        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1)


# ============================================================
#  场景 3：常见激活函数
# ============================================================
class FCN_ActivationFunctions(Scene):
    def construct(self):
        title = Text("常见激活函数", font_size=42, color=YELLOW).to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=0.8)

        # ---- Sigmoid ----
        ax_sig = Axes(
            x_range=[-6, 6, 2], y_range=[-0.2, 1.2, 0.5],
            x_length=3.5, y_length=2.2,
            tips=False,
            axis_config={"include_numbers": False, "stroke_width": 1.5}
        )
        sig_graph = ax_sig.plot(lambda x: 1 / (1 + np.exp(-x)), color=BLUE)
        sig_label = MathTex(r"\sigma(x) = \frac{1}{1+e^{-x}}", font_size=26, color=BLUE)
        sig_title = Text("Sigmoid", font_size=22, color=BLUE)
        sig_group = VGroup(sig_title, VGroup(ax_sig, sig_graph), sig_label).arrange(DOWN, buff=0.2)

        # ---- ReLU ----
        ax_relu = Axes(
            x_range=[-4, 4, 2], y_range=[-0.5, 4, 2],
            x_length=3.5, y_length=2.2,
            tips=False,
            axis_config={"include_numbers": False, "stroke_width": 1.5}
        )
        relu_graph = ax_relu.plot(lambda x: max(0, x), color=GREEN)
        relu_label = MathTex(r"\text{ReLU}(x) = \max(0, x)", font_size=26, color=GREEN)
        relu_title = Text("ReLU", font_size=22, color=GREEN)
        relu_group = VGroup(relu_title, VGroup(ax_relu, relu_graph), relu_label).arrange(DOWN, buff=0.2)

        # ---- Tanh ----
        ax_tanh = Axes(
            x_range=[-4, 4, 2], y_range=[-1.2, 1.2, 0.5],
            x_length=3.5, y_length=2.2,
            tips=False,
            axis_config={"include_numbers": False, "stroke_width": 1.5}
        )
        tanh_graph = ax_tanh.plot(lambda x: np.tanh(x), color=RED)
        tanh_label = MathTex(r"\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}", font_size=26, color=RED)
        tanh_title = Text("Tanh", font_size=22, color=RED)
        tanh_group = VGroup(tanh_title, VGroup(ax_tanh, tanh_graph), tanh_label).arrange(DOWN, buff=0.2)

        # 排列三个图
        all_graphs = VGroup(sig_group, relu_group, tanh_group).arrange(RIGHT, buff=0.6)
        all_graphs.next_to(title, DOWN, buff=0.6)

        for group, graph in zip(
            [sig_group, relu_group, tanh_group],
            [sig_graph, relu_graph, tanh_graph]
        ):
            graph_title = group[0]
            axes_group = group[1]
            label = group[2]
            self.play(FadeIn(axes_group[0]), Write(graph_title), run_time=0.5)
            self.play(Create(graph), run_time=1)
            self.play(FadeIn(label, shift=UP * 0.2), run_time=0.5)
            self.wait(0.3)

        # ---- 特性比较 ----
        comparison = VGroup(
            Text("Sigmoid：输出 (0,1)，适合概率输出，存在梯度消失问题", font_size=18, color=BLUE),
            Text("ReLU：计算简单，缓解梯度消失，但可能出现死神经元", font_size=18, color=GREEN),
            Text("Tanh：输出 (-1,1)，零中心化，但同样有梯度消失", font_size=18, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        comparison.to_edge(DOWN, buff=0.4)

        for line in comparison:
            self.play(FadeIn(line, shift=RIGHT * 0.3), run_time=0.5)

        self.wait(2.5)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1)


# ============================================================
#  场景 4：前向传播完整流程（矩阵形式）
# ============================================================
class FCN_ForwardPass(Scene):
    def construct(self):
        title = Text("前向传播 Forward Propagation", font_size=42, color=YELLOW)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.8)

        # ---- 网络结构 ----
        layer_sizes = [3, 4, 2]
        colors = [GREEN_C, BLUE_C, RED_C]
        layers, connections, network = create_network(layer_sizes, colors, spacing=2.5)
        network.scale(0.55).shift(UP * 1.2 + LEFT * 3.5)

        layer_labels_text = ["输入层", "隐藏层", "输出层"]
        layer_labels = VGroup()
        for i, txt in enumerate(layer_labels_text):
            lab = Text(txt, font_size=16, color=colors[i]).next_to(layers[i], DOWN, buff=0.3)
            layer_labels.add(lab)

        self.play(FadeIn(network), FadeIn(layer_labels), run_time=0.8)
        self.wait(0.3)

        # ---- 公式区域 ----
        formula_title = Text("矩阵形式的前向传播", font_size=28, color=YELLOW)
        formula_title.next_to(network, RIGHT, buff=0.8).align_to(network, UP)

        # 第一层
        eq1_label = Text("隐藏层计算：", font_size=20, color=BLUE_C)
        eq1 = MathTex(
            r"\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]}",
            font_size=32
        )
        eq1a = MathTex(
            r"\mathbf{a}^{[1]} = \sigma(\mathbf{z}^{[1]})",
            font_size=32
        )
        eq1_group = VGroup(eq1_label, eq1, eq1a).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        eq1_group.next_to(formula_title, DOWN, buff=0.5).align_to(formula_title, LEFT)

        # 第二层
        eq2_label = Text("输出层计算：", font_size=20, color=RED_C)
        eq2 = MathTex(
            r"\mathbf{z}^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]}",
            font_size=32
        )
        eq2a = MathTex(
            r"\hat{\mathbf{y}} = \sigma(\mathbf{z}^{[2]})",
            font_size=32
        )
        eq2_group = VGroup(eq2_label, eq2, eq2a).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        eq2_group.next_to(eq1_group, DOWN, buff=0.5).align_to(eq1_group, LEFT)

        # 通用公式
        general_box = SurroundingRectangle(
            VGroup(eq1, eq1a, eq2, eq2a), color=YELLOW, buff=0.25, stroke_width=1.5
        )

        general_eq = MathTex(
            r"\mathbf{a}^{[l]} = \sigma\!\left(\mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}\right)",
            font_size=34, color=YELLOW
        )
        general_label = Text("通用公式 (第 l 层):", font_size=20, color=YELLOW)
        general_group = VGroup(general_label, general_eq).arrange(DOWN, buff=0.15)
        general_group.to_edge(DOWN, buff=0.5)

        self.play(Write(formula_title), run_time=0.6)

        # 前向传播动画 —— 第一层
        # 信号从输入层到隐藏层
        self.play(
            *[n.animate.set_fill(GREEN, opacity=0.7) for n in layers[0]],
            run_time=0.5
        )
        signal_dots_1 = VGroup()
        for conn_line in connections[0]:
            dot = Dot(color=YELLOW, radius=0.06)
            dot.move_to(conn_line.get_start())
            signal_dots_1.add(dot)

        self.play(LaggedStart(*[FadeIn(d) for d in signal_dots_1], lag_ratio=0.02), run_time=0.3)
        self.play(
            *[d.animate.move_to(connections[0][i].get_end()) for i, d in enumerate(signal_dots_1)],
            run_time=1.0
        )
        self.play(
            *[n.animate.set_fill(BLUE, opacity=0.7) for n in layers[1]],
            FadeOut(signal_dots_1),
            run_time=0.5
        )

        self.play(FadeIn(eq1_group, shift=LEFT * 0.3), run_time=0.8)
        self.wait(0.5)

        # 前向传播动画 —— 第二层
        signal_dots_2 = VGroup()
        for conn_line in connections[1]:
            dot = Dot(color=YELLOW, radius=0.06)
            dot.move_to(conn_line.get_start())
            signal_dots_2.add(dot)

        self.play(LaggedStart(*[FadeIn(d) for d in signal_dots_2], lag_ratio=0.02), run_time=0.3)
        self.play(
            *[d.animate.move_to(connections[1][i].get_end()) for i, d in enumerate(signal_dots_2)],
            run_time=1.0
        )
        self.play(
            *[n.animate.set_fill(RED, opacity=0.7) for n in layers[2]],
            FadeOut(signal_dots_2),
            run_time=0.5
        )

        self.play(FadeIn(eq2_group, shift=LEFT * 0.3), run_time=0.8)
        self.wait(0.5)

        # 通用公式
        self.play(Create(general_box), run_time=0.5)
        self.play(Write(general_label), Write(general_eq), run_time=1.2)

        self.wait(2.5)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1)


# ============================================================
#  场景 5：损失函数
# ============================================================
class FCN_LossFunction(Scene):
    def construct(self):
        title = Text("损失函数 Loss Function", font_size=42, color=YELLOW)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=0.8)

        # MSE
        mse_title = Text("均方误差 (MSE) — 回归任务", font_size=26, color=GREEN_C)
        mse_eq = MathTex(
            r"\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2",
            font_size=38
        )
        mse_group = VGroup(mse_title, mse_eq).arrange(DOWN, buff=0.3)

        # Cross-Entropy
        ce_title = Text("交叉熵 (Cross-Entropy) — 分类任务", font_size=26, color=BLUE_C)
        ce_eq = MathTex(
            r"\mathcal{L}_{\text{CE}} = -\frac{1}{n}\sum_{i=1}^{n}"
            r"\left[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\right]",
            font_size=34
        )
        ce_group = VGroup(ce_title, ce_eq).arrange(DOWN, buff=0.3)

        all_loss = VGroup(mse_group, ce_group).arrange(DOWN, buff=0.8)
        all_loss.next_to(title, DOWN, buff=0.7)

        # MSE 动画
        self.play(Write(mse_title), run_time=0.6)
        self.play(Write(mse_eq), run_time=1.2)
        self.wait(0.5)

        # 高亮 y_i - hat{y}_i
        highlight_box = SurroundingRectangle(mse_eq, color=YELLOW, buff=0.1, stroke_width=1.5)
        note1 = Text("预测值与真实值的差的平方的均值", font_size=20, color=GREY_B)
        note1.next_to(mse_eq, DOWN, buff=0.2)
        self.play(Create(highlight_box), FadeIn(note1), run_time=0.6)
        self.wait(0.5)
        self.play(FadeOut(highlight_box), FadeOut(note1), run_time=0.3)

        # Cross-Entropy 动画
        self.play(Write(ce_title), run_time=0.6)
        self.play(Write(ce_eq), run_time=1.5)
        self.wait(0.5)

        note2 = Text("衡量两个概率分布之间的距离", font_size=20, color=GREY_B)
        note2.next_to(ce_eq, DOWN, buff=0.2)
        self.play(FadeIn(note2), run_time=0.5)

        # 目标
        goal = VGroup(
            Text("训练目标：", font_size=24, color=YELLOW),
            MathTex(r"\min_{\mathbf{W}, \mathbf{b}} \; \mathcal{L}(\mathbf{W}, \mathbf{b})", font_size=36),
        ).arrange(RIGHT, buff=0.3)
        goal.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(goal, shift=UP * 0.3), run_time=0.8)

        self.wait(2.5)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1)


# ============================================================
#  场景 6：反向传播 & 梯度下降
# ============================================================
class FCN_Backpropagation(Scene):
    def construct(self):
        title = Text("反向传播 & 梯度下降", font_size=42, color=YELLOW)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.8)

        # ---- 网络（简化版） ----
        layer_sizes = [2, 3, 2]
        colors = [GREEN_C, BLUE_C, RED_C]
        layers, connections, network = create_network(layer_sizes, colors, spacing=2.2)
        network.scale(0.5).shift(UP * 1.5)

        self.play(FadeIn(network), run_time=0.6)

        # ---- 反向传播信号 ----
        bp_title = Text("误差反向传递（链式法则）", font_size=24, color=RED_C)
        bp_title.next_to(network, DOWN, buff=0.3)
        self.play(Write(bp_title), run_time=0.5)

        # 红色信号从右向左
        for conn_group in reversed(connections):
            signal_dots = VGroup()
            for conn_line in conn_group:
                dot = Dot(color=RED, radius=0.06)
                dot.move_to(conn_line.get_end())
                signal_dots.add(dot)
            self.play(LaggedStart(*[FadeIn(d) for d in signal_dots], lag_ratio=0.02), run_time=0.2)
            self.play(
                *[d.animate.move_to(conn_group[i].get_start()) for i, d in enumerate(signal_dots)],
                run_time=0.8
            )
            self.play(FadeOut(signal_dots), run_time=0.2)

        self.wait(0.3)

        # ---- 链式法则公式 ----
        chain_title = Text("链式法则 Chain Rule", font_size=26, color=YELLOW)
        chain_eq = MathTex(
            r"\frac{\partial \mathcal{L}}{\partial w^{[l]}} = "
            r"\frac{\partial \mathcal{L}}{\partial a^{[l]}} \cdot "
            r"\frac{\partial a^{[l]}}{\partial z^{[l]}} \cdot "
            r"\frac{\partial z^{[l]}}{\partial w^{[l]}}",
            font_size=32
        )
        chain_group = VGroup(chain_title, chain_eq).arrange(DOWN, buff=0.25)
        chain_group.next_to(bp_title, DOWN, buff=0.4)

        self.play(Write(chain_title), run_time=0.5)
        self.play(Write(chain_eq), run_time=1.5)
        self.wait(0.5)

        # ---- 梯度下降更新规则 ----
        gd_title = Text("梯度下降更新规则", font_size=26, color=YELLOW)
        gd_eq_w = MathTex(
            r"\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}",
            font_size=34
        )
        gd_eq_b = MathTex(
            r"\mathbf{b}^{[l]} \leftarrow \mathbf{b}^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}}",
            font_size=34
        )
        eta_note = MathTex(r"\text{ : 学习率 (learning rate)}", font_size=26, color=GREY_B)

        gd_group = VGroup(gd_title, gd_eq_w, gd_eq_b, eta_note).arrange(DOWN, buff=0.2)
        gd_group.to_edge(DOWN, buff=0.4)

        self.play(Write(gd_title), run_time=0.5)
        self.play(Write(gd_eq_w), run_time=1)
        self.play(Write(gd_eq_b), run_time=1)
        self.play(FadeIn(eta_note), run_time=0.5)

        self.wait(2.5)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1)


# ============================================================
#  场景 7：完整训练流程总结
# ============================================================
class FCN_TrainingSummary(Scene):
    def construct(self):
        title = Text("全连接网络训练流程总结", font_size=42, color=YELLOW)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=0.8)

        # ---- 流程图 ----
        steps_data = [
            ("1. 初始化", "随机初始化 W, b", BLUE),
            ("2. 前向传播", r"计算 $\hat{y} = f(x; W, b)$", GREEN),
            ("3. 计算损失", r"$\mathcal{L}(y, \hat{y})$", ORANGE),
            ("4. 反向传播", r"计算 $\frac{\partial \mathcal{L}}{\partial W}$", RED),
            ("5. 更新参数", r"$W \leftarrow W - \eta \nabla \mathcal{L}$", PURPLE),
            ("6. 重复迭代", "直到损失收敛", YELLOW),
        ]

        step_groups = VGroup()
        for step_title, step_desc, color in steps_data:
            box = RoundedRectangle(
                corner_radius=0.15, width=4.5, height=0.9,
                color=color, fill_opacity=0.15, stroke_width=2
            )
            t = Text(step_title, font_size=22, color=color, weight=BOLD)
            d = Text(step_desc, font_size=16, color=GREY_B)
            content = VGroup(t, d).arrange(DOWN, buff=0.05)
            content.move_to(box)
            step_groups.add(VGroup(box, content))

        # 排列为两列
        left_col = VGroup(step_groups[0], step_groups[1], step_groups[2]).arrange(DOWN, buff=0.3)
        right_col = VGroup(step_groups[3], step_groups[4], step_groups[5]).arrange(DOWN, buff=0.3)
        flow = VGroup(left_col, right_col).arrange(RIGHT, buff=0.8)
        flow.next_to(title, DOWN, buff=0.6)

        # 添加箭头
        arrows = VGroup()
        order = [
            step_groups[0], step_groups[1], step_groups[2],
            step_groups[3], step_groups[4], step_groups[5]
        ]
        for i in range(len(order) - 1):
            start_box = order[i][0]
            end_box = order[i + 1][0]
            if i == 2:  # 从左列底部到右列顶部
                arrow = CurvedArrow(
                    start_box.get_right(), end_box.get_left(),
                    color=WHITE, stroke_width=2
                )
            elif i < 2:
                arrow = Arrow(
                    start_box.get_bottom(), end_box.get_top(),
                    color=WHITE, stroke_width=2, buff=0.1
                )
            else:
                arrow = Arrow(
                    start_box.get_bottom(), end_box.get_top(),
                    color=WHITE, stroke_width=2, buff=0.1
                )
            arrows.add(arrow)

        # 循环箭头（从步骤6回到步骤2）
        loop_arrow = CurvedArrow(
            step_groups[5][0].get_right() + RIGHT * 0.1,
            step_groups[1][0].get_right() + RIGHT * 0.1,
            color=YELLOW, stroke_width=2.5, angle=-TAU / 3
        )
        loop_label = Text("Epoch", font_size=18, color=YELLOW)
        loop_label.next_to(loop_arrow, RIGHT, buff=0.15)

        # 逐步动画
        for i, step in enumerate(order):
            self.play(FadeIn(step, shift=UP * 0.2), run_time=0.5)
            if i < len(arrows):
                self.play(GrowArrow(arrows[i]), run_time=0.3)
            self.wait(0.2)

        self.play(Create(loop_arrow), FadeIn(loop_label), run_time=0.8)

        self.wait(1)

        # ---- 最终总结公式 ----
        summary_box = RoundedRectangle(
            corner_radius=0.2, width=12, height=1.2,
            color=YELLOW, fill_opacity=0.1, stroke_width=2
        ).to_edge(DOWN, buff=0.3)

        summary_eq = MathTex(
            r"\text{输入 } \mathbf{x} \xrightarrow{\mathbf{W}^{[1]}, \mathbf{b}^{[1]}}"
            r" \mathbf{a}^{[1]} \xrightarrow{\mathbf{W}^{[2]}, \mathbf{b}^{[2]}}"
            r" \cdots \xrightarrow{\mathbf{W}^{[L]}, \mathbf{b}^{[L]}}"
            r" \hat{\mathbf{y}} \rightarrow \mathcal{L}",
            font_size=30
        )
        summary_eq.move_to(summary_box)

        self.play(FadeIn(summary_box), Write(summary_eq), run_time=1.5)

        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)