import torch
import numpy as np # 引入库函数
import matplotlib.pyplot as plt
import time

x_data=[1.0 ,2.0 , 3.0 , 4.0 , 5.0]
y_data=[2.0 , 4.0 , 6.0 ,8.0 , 10.0]

w = torch.tensor([1.0])
w.requires_grad = True

def forward(x):
    return x*w

def loss(x,y):
    y_hat= forward(x)
    return (y_hat-y)**2

MSE_list = []
epoch_list = []
weight_list = []

plt.ion()
fig,ax = plt.subplots(figsize = (10,6)) # 创建图表对象和坐标轴对象

plt.pause(10)

for epoch in range(0 , 16 , 1): # 轮数为15轮
    for x,y in zip(x_data , y_data):
        l =loss(x ,y)
        l.backward() # 反向传播
        w.data = w.data - 0.01 * w.grad.data

        w.grad.data.zero_()
    print("progress:", epoch, l.item())
    MSE_list.append(l.item())
    epoch_list.append(epoch)

    # 更新图表
    ax.clear()
    ax.plot(epoch_list , MSE_list , 'b-o' , linewidth = 2 , markersize = 4)
    ax.set_xlabel('epoch' ,fontsize = 12)
    ax.set_ylabel('MSE loss' , fontsize = 12)
    ax.set_title(f'Training progress (Epoch{epoch})' , fontsize = 14)
    ax.grid(True , alpha = 0.3)
    plt.pause(0.5) #每一轮后暂停0.5秒

plt.ioff()
plt.show()


print("predict (after training)", 4, forward(4).item())

# 最终绘制完整的损失曲线
plt.figure(figsize=(10, 6))
plt.plot(epoch_list, MSE_list, 'b-o', linewidth=2, markersize=6)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.title('Final Training Loss Curve', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()



