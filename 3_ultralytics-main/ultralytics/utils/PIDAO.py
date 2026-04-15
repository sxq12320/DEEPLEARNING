import math
import torch
import torch.nn as nn
from collections import deque
from functools import partial
import torch.optim as optim
from torch.optim.optimizer import Optimizer
# 假设从 ultralytics 的相关模块中导入以下必要工具
# from ultralytics.utils import LOGGER, colorstr
# from ultralytics.nn.tasks import unwrap_model
# from ultralytics.utils.torch_utils import MuSGD  # 若有

# ==========================================
# 1. 核心优化器：多通道高阶 PID (PIDAO)
# ==========================================
class PIDAO(Optimizer):
    def __init__(self, params, lr=1e-3, eq_momentum=0.9, kp=None, ki=1.0, kd_channels=None):
        """
        多通道高阶 PID 优化器
        :param kd_channels: list, 包含各个阶数微分通道的系数 [1阶微分系数, 2阶微分系数, ...]
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if kd_channels is None:
            kd_channels = [0.1]  # 默认退化为标准 PID

        defaults = dict(lr=lr, eq_momentum=eq_momentum, kp=kp, ki=ki, kd_channels=kd_channels)
        super(PIDAO, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if group.get('weight_decay', 0) != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                state = self.state[p]
                kd_channels = group['kd_channels']
                num_d_channels = len(kd_channels)

                # 初始化状态字典
                if len(state) == 0:
                    state['step'] = 0
                    state['I'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # 使用 deque 存储所需的历史梯度，最大长度取决于我们需要的最高阶导数 (N阶需要 N+1 个历史状态)
                    state['grad_history'] = deque(maxlen=num_d_channels + 1)

                state['step'] += 1
                I = state['I']
                grad_hist = state['grad_history']

                # 将当前梯度插入历史队列的最左端 (索引 0 永远是 g_t)
                grad_hist.appendleft(grad.clone())

                # 1. 积分通道 (I)
                I.add_(grad)

                # 2. 比例通道 (P) -> 如果 Kp 是 None，则默认使用 1.0
                kp = group['kp'] if group['kp'] is not None else 1.0
                update = torch.mul(grad, kp)
                update.add_(I, alpha=group['ki'])

                # 3. 高阶微分多通道 (D)
                # 使用数学推导中的二项式系数计算 N 阶差分
                for k, kd in enumerate(kd_channels):
                    order = k + 1  # 当前的导数阶数 (1阶, 2阶...)
                    
                    # 如果历史梯度数量不足以计算当前阶数的差分，则跳过该通道
                    if len(grad_hist) < order + 1:
                        continue
                    
                    diff_k = torch.zeros_like(grad)
                    for j in range(order + 1):
                        # (-1)^j * C(order, j) * g_{t-j}
                        coef = ((-1) ** j) * math.comb(order, j)
                        diff_k.add_(grad_hist[j], alpha=coef)
                    
                    # 将该微分通道的结果叠加到总更新量中
                    update.add_(diff_k, alpha=kd)

                # 4. 应用最终更新
                p.add_(update, alpha=-group['lr'])

        return loss