import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F

class PIDAO(Optimizer):
    '''
    using :
        一个全新的优化器，PIDAO
    Args:
        params : 待优化的参数迭代器
        lr : 学习率
        a : 阻尼系数
        kp：比例系数
        ki：积分系数
        kd：微分系数
    Returns：

    '''
    def __init__(self , params , lr = 0.01 , a = 11.11 , kp = 111.11 , ki = 1 , kd = 0.1):
        defaults = dict(lr=lr, a=a, kp=kp, ki=ki, kd=kd)# 超参数存到字典里面
        super(PIDAO , self).__init__(params , defaults)
    @ torch.no_grad()
    def step(self , closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            h = group['lr']
            a = group['a']
            kp = group['kp']
            ki = group['ki']
            kd = group['kd']

            denom = 1.0 + a * h
            grad_coeff = h * (kp-a*kd)

            for p in group['params']:
                if p.grad is None:
                    continue

                # 获取当前梯度 ∇f(x_k)
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0

                    state['z'] = torch.zeros_like(p , memory_format=torch.preserve_format)
                    state['y'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                z = state['z']
                y = state['y']
                state['step'] += 1

                z.add_(grad, alpha=h)
                y.sub_(grad_coeff, alpha=grad_coeff)
                y.sub_(z, alpha=h * ki)
                y.div_(denom)
                p.add_(y, alpha=h)
                p.sub_(grad, alpha=h * kd)
        return loss
