import torch
import matplotlib.pyplot as plt
import numpy as np
import math

x = torch.arange(4.0) #定义向量x
x.requires_grad_(True) #开始存储梯度
x.grad #访问梯度

y = 2*torch.dot(x,x) #定义函数y = 2 xT x
y.backward() #求导、反向传递

Text = x.grad ==4*x

x.grad.zero_() #清除之前的梯度缓存
y = x.sum() #重新建立一个新的函数
#print(y)
y.backward() #求导、反向传递
GRAD_x = x.grad #访问梯度
#print(GRAD_x)

x.grad.zero_() #清除之前的梯度缓存
y=x*x
# 等价于 y.backward(torch.ones(len(x)))
y.sum().backward() #求导、反向传递
GRAD_x = x.grad
# print(GRAD_x)

x.grad.zero_()
y = x* x #对应元素相乘
print(y)
u = y.detach() #u就相当于一个常量 并不是变量
z = u * x
z.sum().backward() #求导、反向传递
GRAD_x = x.grad
Test = x.grad == u #检验
print(GRAD_x)
print(Test)