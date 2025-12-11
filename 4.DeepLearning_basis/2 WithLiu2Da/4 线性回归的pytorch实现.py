import numpy as np
import matplotlib.pyplot as plt
import torch 

# 1 训练数据准备
x_data = torch.tensor([[1.0] , [2.0] , [3.0]])
y_data = torch.tensor([[2.0] , [4.0] , [6.0]])

# 2 设计模型 线性模型 y=x*w+b loss 求和平方函数
class LinearModel(torch.nn.Module): 
    def __init__(self):
        super(LinearModel , self).__init__()
        self.linear = torch.nn.Linear(1,1) # 定义线性模型 y = w*x + b

    def forward(self , x):
        y_pred = self.linear(x) # 直节调用self.linear()这个方法
        return y_pred
    
Model = LinearModel() # 模型进行实例化

# 3 定义损失函数并确定优化器
criterion = torch.nn.MSELoss(size_average=False) # sum
optimizer = torch.optim.SGD(Model.parameters(),lr =0.05)

# 4 开始进行学习
loss_list = []
epoch_list = []
w_list = []
b_list = []

for epoch in range(0 , 51 , 1):
    epoch_list.append(epoch)
    y_pred = Model(x_data)
    loss = criterion(y_pred , y_data)
    loss_list.append(loss.item()) # 转换成标量
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    w_list.append(Model.linear.weight.item())
    b_list.append(Model.linear.bias.item())


for e,l,w ,b in zip(epoch_list , loss_list , w_list , b_list):
    print(f'第{e}轮的时候，损失为{l:.6f}，权重为{w:.6f} , 偏置为{b:.6f} \n')

fig , axes = plt.subplots(3,1 , figsize =(5, 6))

axes[0].plot(epoch_list , loss_list , 'b-' , linewidth = 2 , label = 'Loss curve')
axes[0].set_xlabel('Number of training rounds')
axes[0].set_ylabel('Size of loss')
axes[0].set_title('Loss curve')
axes[0].legend()


axes[1].plot(epoch_list,w_list,'b-',linewidth=2 , label = 'Weight curve')
axes[1].set_xlabel('Number of training rounds')
axes[1].set_ylabel('Size of Weight')
axes[1].legend()

axes[2].plot(epoch_list,b_list,'r-',linewidth=2 , label = 'Bias curves')
axes[2].set_xlabel('Number of training rounds')
axes[2].set_ylabel('Size of Bias')
axes[2].legend()

plt.tight_layout()
plt.show()



    
