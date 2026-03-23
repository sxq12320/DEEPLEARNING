import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm  # <--- 引入进度条库
from data import *
from net import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 在GPU上跑

weight_path = 'params/unet.path'
data_path = r'E:\mastercode\data\VOC\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'
save_path = 'train_images'

def train_mine(EPOCHS:int):
    data_loader = DataLoader(MyDataset(data_path) , batch_size = 1 , shuffle = True)
    net = unet()
    net = net.to(device)

    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print("the weight path is existed")
    else:
        print("the weight path is not existed")


    optimizer = optim.AdamW(net.parameters(), lr = 0.01)
    loss_func = nn.BCELoss()

    for epoch in range(EPOCHS):
        for i , (image, segment_iamge) in enumerate(data_loader):
            image = image.to(device)
            segment_iamge = segment_iamge.to(device)

            out_image = torch.sigmoid(net(image))
            train_loss = loss_func(out_image, segment_iamge)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            if epoch % 1 == 0 :
                print(f'{epoch} -- train_loss=====>: {train_loss.item()}')

            if i % 50 == 0 :
                torch.save(net.state_dict(), weight_path)

            _img = image[0]
            _seg = segment_iamge[0]
            _out_img = out_image[0]

            img = torch.stack([_img,_seg,_out_img] , dim=0)
            save_image(img ,f'{save_path}/{i}.png')


def train(EPOCHS: int):
    data_loader = DataLoader(MyDataset(data_path), batch_size=1, shuffle=True)
    net = unet().to(device)

    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print(f"[*] The weight path exists, loaded weights from {weight_path}")
    else:
        print("[!] The weight path does not exist, starting from scratch")

    optimizer = optim.AdamW(net.parameters(), lr=0.01)
    # 💡 友情提示：上个问题中提到的，更推荐使用 BCEWithLogitsLoss() 替代 BCELoss() + sigmoid
    loss_func = nn.BCELoss()

    # 模仿 YOLO 的表头打印
    print(f"\n{'Epoch':>10} {'GPU_mem':>10} {'Train_Loss':>12} {'Learn_Rate':>12} {'Img_Size':>10}")

    for epoch in range(EPOCHS):
        net.train()  # 养成好习惯，声明训练模式

        # 使用 tqdm 包装 data_loader，生成动态进度条
        pbar = tqdm(enumerate(data_loader), total=len(data_loader), bar_format='{l_bar}{bar:10}{r_bar}')

        for i, (image, segment_iamge) in pbar:
            image = image.to(device)
            segment_iamge = segment_iamge.to(device)

            # 前向传播与反向传播
            optimizer.zero_grad()  # 建议放到 forward 之前
            out_image = torch.sigmoid(net(image))
            train_loss = loss_func(out_image, segment_iamge)
            train_loss.backward()
            optimizer.step()

            # --- 👇 收集 YOLO 风格需要展示的参数 👇 ---

            # 1. 计算当前 GPU 显存占用 (单位 GB)
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'

            # 2. 获取当前的学习率
            current_lr = optimizer.param_groups[0]['lr']

            # 3. 图像尺寸
            img_size = f"{image.shape[2]}x{image.shape[3]}"

            # 4. 格式化输出字符串 (严格对齐)
            desc = (f"{f'{epoch + 1}/{EPOCHS}':>10} "
                    f"{mem:>10} "
                    f"{train_loss.item():>12.4f} "
                    f"{current_lr:>12.6f} "
                    f"{img_size:>10}")

            # 动态更新到进度条的最左侧
            pbar.set_description(desc)

            # --- 👆 参数收集结束 👆 ---

            # 定期保存权重 (这里你原来是 i % 50，我没改。建议按 epoch 保存更好)
            if epoch % 1 == 0:
                torch.save(net.state_dict(), weight_path)

            # 保存图片 (为了不拖慢训练速度，可以考虑降低保存频率，比如 if i % 100 == 0)
            if i == 0 or i == 10 or i== 50 or i==100 or i== 150 or i==200:
                _img = image[0]
                _seg = segment_iamge[0]
                _out_img = out_image[0]
                img = torch.stack([_img, _seg, _out_img], dim=0)
                save_image(img, f'{save_path}/{epoch}_{i}.png')



if __name__ == '__main__':
    EPOCHS = 10
    train(EPOCHS)