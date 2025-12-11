import numpy as np
import matplotlib.pyplot as plt
import math

def target_f(x):
    """定义任务一的目标函数
    Args:
        x: 自变量
    Returns:
        float: 函数的因变量
    """
    return 2*x*(1-x+2*x*x)*np.exp(-(x*x)/2)

def train_data(train_size,noise_std = 0.1):
    """生成训练样本
    Args:
        train_size: 训练样本的个数
        nosize_std: 正态分布的标准差
    Returns:
        x_train: float 自变量数据
        y_train: float 应变量数据    
    """
    x_train = np.linspace(-4 , 4 , train_size)
    noise = np.random.normal(0 , noise_std , size=x_train.shape)
    y_train=target_f(x_train)+noise
    return x_train , y_train

def test_data(test_size=160):
    """生成检验样本
    Args:
        test_size: 检验样本数量
    Returns:
        x_test: float 自变量数据
        y_test: float 应变量数据
    """
    x_test = np.linspace(-4 , 4 , test_size)
    y_test = target_f(x_test)
    return x_test , y_test

def sigmoid(x):
    """定义激活函数
    Args:
        x: 激活函数sigmoid的自变量
    Returns:
        float: 激活函数的应变量
    """
    return 1/(1+np.exp(-x))

def sigmoid_dao(x):
    s = sigmoid(x)
    return s*(1-s)

class FirstMLP:
    def __init__(self,input_size=1,hidden_size=10,output_size=1):
        self.weights1 = np.random.randn(input_size,hidden_size)*0.1
        self.bias1 = np.zeros((1,hidden_size))
        self.weights2 = np.random.randn(hidden_size,output_size)*0.1
        self.bias2 = np.zeros((1,output_size))

    def forward(self,x):
        self.z1 = np.dot(x,self.weights1)+self.bias1
        self.a1 = sigmoid(self.z1)

        self.z2 = np.dot(self.a1 , self.weights2)+self.bias2
        self.a2 = self.z2

        return self.a2
    
    def backward(self,x ,y , output , learning_rate):
        output_error = y-output
        output_delta = output_error

        hidden_error = output_delta.dot(self.weights2.T)
        hidden_delta = hidden_error * sigmoid_dao(self.z1)

        self.weights2 += self.a1.T.dot(output_delta)*learning_rate
        self.bias2 += np.sum(output_delta , axis=0 , keepdims=True)*learning_rate
        self.weights1 += x.T.dot(hidden_delta) * learning_rate
        self.bias1 += np.sum(hidden_delta , axis=0 , keepdims=True)*learning_rate
        return np.mean(np.square(output_error))
    
def train_model(model, x_train, y_train, epochs=10000, learning_rate=0.01):
    """训练MLP模型：通过多轮前向+反向传播优化参数"""
    # 调整输入形状为二维数组 (样本数, 特征数)，适应矩阵运算
    x_train = x_train.reshape(-1, 1)  # -1表示自动计算样本数，1表示1个特征
    y_train = y_train.reshape(-1, 1)  # 标签同样调整为二维数组
    
    losses = []  # 存储每轮的损失值，用于后续可视化
    for i in range(epochs):  # 迭代训练epochs轮
        output = model.forward(x_train)  # 前向传播：计算当前预测值
        # 反向传播：根据误差更新参数，并获取当前损失
        loss = model.backward(x_train, y_train, output, learning_rate)
        if loss<=0.01:
            break
        losses.append(loss)  # 记录损失值
        # 每100轮打印一次损失，监控训练进度
        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss:.6f}")  # .6f表示保留6位小数
    return losses  # 返回训练过程的损失列表

x_train , y_train = train_data(train_size=80)
x_test , y_test = test_data()

# 创建模型：隐藏层15个神经元
model = FirstMLP(hidden_size=10)
# 训练模型：20000轮，学习率0.005
losses = train_model(model, x_train, y_train, epochs=20000, learning_rate=0.01)

# 模型预测：用训练好的模型预测测试数据
y_pred = model.forward(x_test.reshape(-1, 1)).flatten()  # 展平为一维数组，便于绘图


# # 第一个子图：训练损失曲线
# plt.subplot(1, 2, 1)  # 1行2列的第1个图
# plt.plot(losses)  # 绘制损失随轮次的变化
# plt.title('Training Loss')  # 标题：训练损失
# plt.xlabel('Epoch')  # x轴标签：轮次
# plt.ylabel('MSE')  # y轴标签：均方误差
# plt.grid(True, linestyle='--', alpha=0.7)  # 显示网格线，虚线，透明度0.7

# 第二个子图：函数逼近效果对比
# plt.subplot(1, 2, 2)  # 1行2列的第2个图
plt.figure(figsize=(8,6))
plt.plot(x_test, y_test, 'b-', label='Original Function')  # 绘制真实函数（蓝色实线）
plt.scatter(x_train, y_train, c='orange', label='Training Data (with noise)')  # 绘制带噪声的训练数据（橙色点）
plt.plot(x_test, y_pred, 'r--', label='MLP Prediction')  # 绘制模型预测结果（红色虚线）
plt.title('Function Approximation')  # 标题：函数逼近效果
plt.xlabel('x')  # x轴标签
plt.ylabel('f(x)')  # y轴标签
plt.legend()  # 显示图例
plt.grid(True, linestyle='--', alpha=0.7)  # 显示网格线

plt.tight_layout()  # 自动调整子图间距，避免重叠
plt.show()  # 显示图像

# # plt.subplot(2,2,1)  
# plt.plot(x, y_original, color='blue', linewidth=3, label='origin f(x)')
# plt.plot(x, y_noisy, color='red', linewidth=1,label='with noise f(x)')  
# plt.xlabel('x', fontsize=12)
# plt.ylabel('f(x)', fontsize=12)
# plt.legend()
# plt.title("input f(x)")
# plt.grid(True, linestyle='--', alpha=0.5)  

# plt.show()