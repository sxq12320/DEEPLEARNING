import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0 , 2.0 , 3.0 , 4.0 , 5.0]
y_data = [2.0 , 4.0 , 6.0 , 8.0 , 10.0]

def forward(x):
    return x * w

def MSE_loss(x , y):
    y_hat = forward(x)
    res = (y_hat -  y)**(2)
    return res

w_list = []
mse_loss = []

for w in np.arange(0.0 , 4.1 , 0.1):
    w_list.append(w)
    sum = 0
    for x , y in zip(x_data , y_data):
        sum += MSE_loss(x , y)
    sum = sum/(len(x_data))
    mse_loss.append(sum)

plt.plot(w_list , mse_loss)
plt.scatter(w_list , mse_loss , color = 'r' ,s = 20 )
plt.xlabel('W')
plt.ylabel('MSE')
plt.show()

print(w_list)
print(mse_loss)


    

