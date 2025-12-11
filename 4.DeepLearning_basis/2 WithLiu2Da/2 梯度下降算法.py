'''
@name:  梯度下降算法
'''

import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0 , 2.0 , 3.0 , 4.0]
y_data = [2.0 , 4.0 , 6.0 , 8.0]

global w,alpha #design global variables
w = 1.0
alpha = 0.01 # learning rate

def forward(x):
    return x*w

def cost(xs , ys):
    res = 0
    for x,y in zip(xs , ys):
        res += (x*w - y)**2
    res = res /len(xs)
    return res

def gradient(xs , ys):
    sum = 0
    for x , y in zip(xs , ys):
        sum = 2*x*(x*w-y)
    sum = sum/len(xs)
    return sum 

epc_list = []
cos_list = []

for epoch in range(0 , 100 , 1):
    epc_list.append(epoch)
    cost_val = cost(x_data , y_data)
    cos_list.append(cost_val)
    grad_val = gradient(x_data , y_data)
    w -= alpha * grad_val

plt.plot(epc_list , cos_list , 'r')
plt.xlabel('epoch')
plt.ylabel('MSE')

plt.show()
    
