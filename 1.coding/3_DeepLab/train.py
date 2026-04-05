import torch
import torch.nn as nn
import torch.nn.functional as F
from net import BackBone_VGG_16 , BackBone_VGG_16_MSC
import os

NUM_CLASSES = 10
LR = 0.001
MOMENTUM = 0.9
EPOCHS = 100

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = BackBone_VGG_16_MSC(num_classes=NUM_CLASSES).to(device)


optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
loss_func = nn.CrossEntropyLoss(ignore_index=255)
weight_path = r"3_DeepLab\\weights"


model.train()
if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path))
            print("the model weights are loaded")
else:
    print("none model weights")

for epoch in range(EPOCHS):
    for images , masks in train_loader:
        img = images.to_device(device)
        mask = masks.to_device(device)
        
        out_image = model(img)
        train_loss = loss_func(out_image , mask)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        pred_vis = out_image[0].argmax(dim = 0).float() / 29.0

        if epoch % 1 ==0:
              print(f'{epoch} -- train loss: {train_loss.item()}')

        
        

        

