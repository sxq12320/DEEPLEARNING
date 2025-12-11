import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  
plt.rcParams['axes.unicode_minus'] = False 

#### 1. 目标函数（不变）####
def target_function(x1, x2):
    '''定义目标函数（处理x=0的奇点）'''
    x1 = np.where(np.abs(x1) < 1e-6, 1e-6, x1)
    x2 = np.where(np.abs(x2) < 1e-6, 1e-6, x2)
    return ((np.sin(x1)/x1) * (np.sin(x2)/x2))

#### 2. 数据生成（新增输入归一化）####
def train_data(train_size=11):
    # 生成[-10,10]的原始数据
    x1_train = np.linspace(-10, 10, train_size)  
    x2_train = np.linspace(-10, 10, train_size)
    X1_train, X2_train = np.meshgrid(x1_train, x2_train)
    
    # 关键：输入归一化（缩放到[-1,1]）
    X1_train_norm = X1_train / 10  # 范围[-10,10] → [-1,1]
    X2_train_norm = X2_train / 10
    
    # 拼接特征（用归一化后的数据）
    X_train = np.hstack([X1_train_norm.reshape(-1,1), X2_train_norm.reshape(-1,1)])
    Y_train = target_function(X1_train, X2_train).reshape(-1, 1)  # 标签无需归一化
    return X_train, Y_train, X1_train, X2_train  # 返回原始x用于绘图

def test_data(test_size=21):
    x1_test = np.linspace(-10, 10, test_size)  
    x2_test = np.linspace(-10, 10, test_size)
    X1_test, X2_test = np.meshgrid(x1_test, x2_test)
    
    X1_test_norm = X1_test / 10
    X2_test_norm = X2_test / 10
    
    X_test = np.hstack([X1_test_norm.reshape(-1,1), X2_test_norm.reshape(-1,1)])
    Y_test = target_function(X1_test, X2_test).reshape(-1, 1)
    return X_test, Y_test, X1_test, X2_test

#### 3. 激活函数（替换为ReLU）####
def relu(x):  # ReLU激活函数
    return np.maximum(0, x) 

def relu_dao(x):  # ReLU导数（x>0时为1，否则为0）
    return (x > 0).astype(float)

#### 4. MLP类（修改激活函数）####
class SecondMLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr):
        self.lr = lr
        # 权重初始化（不变，但ReLU对初始值更友好）
        self.w1 = np.random.randn(input_dim, hidden_dim)*0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.w2 = np.random.randn(hidden_dim, output_dim)*0.01
        self.b2 = np.zeros((1, output_dim))        

    def forward(self, x):
        '''前向传播（用ReLU替代Sigmoid）'''
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = relu(self.z1)  # 隐藏层用ReLU
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.z2  # 回归任务输出层无激活
        return self.a2
    
    def backward(self, x, y_true):
        N = x.shape[0]
        # 梯度计算（导数用ReLU的导数）
        dz2 = (self.a2 - y_true)/N
        dw2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # 关键：用relu_dao替代sigmoid_dao
        dz1 = np.dot(dz2, self.w2.T) * relu_dao(self.z1)
        dw1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # 权重更新（不变）
        self.w1 -= self.lr * dw1
        self.w2 -= self.lr * dw2
        self.b1 -= self.lr * db1
        self.b2 -= self.lr * db2

    def train(self, x, y, EPOCHS, LOSS):
        loss_history = []
        for i in range(EPOCHS):
            y_pred = self.forward(x)
            loss = np.mean((y_pred - y)**2)
            loss_history.append(loss)
            self.backward(x, y)

            # 提前停止条件
            if loss <= LOSS:
                print(f"提前停止：Epoch {i+1}, loss={loss:.6f}")
                break
            # 每500轮打印一次（更直观观察收敛过程）
            if (i+1) % 500 == 0:
                print(f"Epoch {i+1}, loss:{loss:.6f}")
        return loss_history

# ---------------------- 训练与绘图（参数微调） ----------------------
# 初始化模型
mlp = SecondMLP(input_dim=2, hidden_dim=10, output_dim=1, lr=0.5)

# 生成训练数据（含归一化）
X_train, Y_train, X1_train, X2_train = train_data(train_size=11)
# 训练模型（最大轮次20000，目标损失0.001）
loss_history = mlp.train(X_train, Y_train, EPOCHS=200000, LOSS=0.001)

# 生成测试数据（含归一化）
X_test, Y_test, X1_test, X2_test = test_data(test_size=21)
# 测试模型
Y_pred = mlp.forward(X_test)
test_loss = np.mean((Y_pred - Y_test)**2)
print(f"\n测试集损失：{test_loss:.6f}")

# 1. 绘制损失曲线（会明显下降）
plt.figure(figsize=(8, 4))
plt.plot(loss_history, color='#1f77b4', linewidth=1.2)
plt.title("训练损失曲线（MSE）")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(alpha=0.3)
plt.show()

# 2. 绘制3D对比图（目标函数 vs 预测结果）
fig = plt.figure(figsize=(12, 5))
# 目标函数
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X1_test, X2_test, Y_test.reshape(X1_test.shape), 
                         cmap='viridis', alpha=0.8, linewidth=0)

ax1.scatter(
    X1_train,  # 训练点的x1坐标（11x11网格）
    X2_train,  # 训练点的x2坐标（11x11网格）
    target_function(X1_train, X2_train),  # 训练点的函数值（真实值）
    color='red',  # 红色标记，醒目
    s=50,  # 标记尺寸，避免太小看不见
    edgecolors='black',  # 边缘黑色，增强区分度
    label='训练数据点'  # 图例标签
)
ax1.set_title("目标函数：$f(x_1,x_2) = \\frac{\sin x_1}{x_1} \cdot \\frac{\sin x_2}{x_2}$")
ax1.set_xlabel("$x_1$")
ax1.set_ylabel("$x_2$")
ax1.set_zlabel("$f(x_1,x_2)$")
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

# 预测结果
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X1_test, X2_test, Y_pred.reshape(X1_test.shape), 
                         cmap='viridis', alpha=0.8, linewidth=0)
ax2.set_title("MLP预测结果（ReLU+归一化）")
ax2.set_xlabel("$x_1$")
ax2.set_ylabel("$x_2$")
ax2.set_zlabel("预测值")
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()