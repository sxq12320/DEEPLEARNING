import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [4.0, 6.0, 8.0, 10.0, 12.0]

def forward(x, w, b):
    return x * w + b

def loss(x, y, w, b):
    return ((forward(x, w, b) - y) * (forward(x, w, b) - y))

# 生成w和b的网格
w_range = np.arange(0, 4.1, 0.1)
b_range = np.arange(0, 4.1, 0.1)
W, B = np.meshgrid(w_range, b_range)

# 计算每个(w,b)组合的MSE
MSE = np.zeros_like(W)

for i in range(len(w_range)):
    for j in range(len(b_range)):
        w_val = w_range[i]
        b_val = b_range[j]
        total_loss = 0
        for x, y in zip(x_data, y_data):
            total_loss += loss(x, y, w_val, b_val)
        MSE[j, i] = total_loss / len(x_data)  # 注意索引顺序

min_idx = np.unravel_index(np.argmin(MSE, axis=None), MSE.shape)
min_w = w_range[min_idx[1]]
min_b = b_range[min_idx[0]]
min_mse = MSE[min_idx]

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111 , projection='3d')

ax.scatter(min_w, min_b, min_mse, color='red', s=100, 
          label=f'Min MSE: {min_mse:.2f}\nw={min_w:.1f}, b={min_b:.1f}')

surf = ax.plot_surface(W , B , MSE , cmap='viridis',alpha = 0.8 , linewidth = 0 , antialiased = True)
ax.set_xlabel('Weight (w)', fontsize=12)
ax.set_ylabel('Bias (b)', fontsize=12)
ax.set_zlabel('MSE Loss', fontsize=12)
ax.set_title('MSE Loss Surface for Linear Regression y = w*x + b', fontsize=14)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='MSE Value')
ax.legend()

ax.view_init(elev=25, azim=40)            
plt.tight_layout()
plt.show()



print(f"Minimum MSE: {min_mse:.4f} at w={min_w:.1f}, b={min_b:.1f}")