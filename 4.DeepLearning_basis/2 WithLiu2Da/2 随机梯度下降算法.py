'''
@name:  随机梯度下降算法
'''

import numpy as np
import matplotlib.pyplot as plt

def forward(x):
    global w
    return x*w

def loss(x , y):
    return (forward(x) - y)**2

def gradient(x , y):
    return (2*x*(forward(x) - y))

x_data = [1.0 , 2.0 , 3.0 , 4.0]
y_data = [2.0 , 4.0 , 6.0 , 8.0]

w = 1.0
alpha = 0.01 # learning rate

epc_list = []
loss_list = []
for epoch in range(0 , 100 , 1):
    epc_list.append(epoch)
    n = 0.0
    sum = 0.0
    for x,y in zip(x_data ,  y_data):
        n+=1
        grad = gradient(x , y)
        w = w - 0.01 * grad
        l = loss(x , y)
        sum += l
    sum =sum / n
    loss_list.append(sum)

plt.plot(epc_list ,  loss_list , 'r')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
    

