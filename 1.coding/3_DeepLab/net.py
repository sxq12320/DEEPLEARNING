import torch 
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    '''
    using :
        基本的运行块， 用来构建backbone
    Args:
        in_channel : 输入的通道数量
        out_channel : 输出的通道数量
        kernel_size : 卷积核的大小
        padding : 填充大小
        stride : 步长大小
    Returns:    
        nn.Senquential序列
    '''
    def __init__(self , in_channel = 3 , out_channel = 64 , kernel_size = 3 , padding = 1 , stride = 1 , dilation = 1):
        super(BasicBlock , self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channel,
                out_channels = out_channel , 
                kernel_size = kernel_size ,
                stride = stride,
                padding = padding , 
                bias = False , 
                dilation=dilation
                ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self , x):
        x = self.layer(x)
        return x

class BackBone_VGG_16 (nn.Module):
    def __init__(self , num_classes = 10):
        super(BackBone_VGG_16, self).__init__()

        self.layer1 = BasicBlock(in_channel = 3 , out_channel= 64 , kernel_size= 3 , padding = 1 , stride = 1)
        self.layer2 = BasicBlock(in_channel = 64 , out_channel= 64 , kernel_size= 3 , padding = 1 , stride = 1)
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer3 = BasicBlock(in_channel=64, out_channel=128, kernel_size=3, padding=1, stride=1)
        self.layer4 = BasicBlock(in_channel=128, out_channel=128, kernel_size=3, padding=1, stride=1)
        self.max2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer5 = BasicBlock(in_channel=128, out_channel=256, kernel_size=3, padding=1, stride=1)
        self.layer6 = BasicBlock(in_channel=256, out_channel=256, kernel_size=3, padding=1, stride=1)
        self.layer7 = BasicBlock(in_channel=256, out_channel=256, kernel_size=3, padding=1, stride=1)
        self.max4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer8 = BasicBlock(in_channel=256, out_channel=512, kernel_size=3, padding=1, stride=1)
        self.layer9 = BasicBlock(in_channel=512, out_channel=512, kernel_size=3, padding=1, stride=1)
        self.layer10 = BasicBlock(in_channel=512, out_channel=512, kernel_size=3, padding=1, stride=1)
        self.max5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer11 = BasicBlock(in_channel=512, out_channel=512, kernel_size=3, padding=2, stride=1 ,dilation=2)
        self.layer12 = BasicBlock(in_channel=512, out_channel=512, kernel_size=3, padding=2, stride=1,dilation=2)
        self.layer13 = BasicBlock(in_channel=512, out_channel=512, kernel_size=3, padding=2, stride=1,dilation=2)
        self.max6 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)


        self.layer14 = BasicBlock(in_channel=512, out_channel=512, kernel_size=3, padding=2, stride=1, dilation=2)
        self.layer15 = BasicBlock(in_channel=512, out_channel=512, kernel_size=3, padding=2, stride=1, dilation=2)
        self.layer16 = BasicBlock(in_channel=512, out_channel=512, kernel_size=3, padding=2, stride=1, dilation=2)
        self.max7 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avg1 = nn.AvgPool2d(kernel_size=3 , stride=1 , padding=1)

        self.layer17 = BasicBlock(in_channel=512, out_channel=1024, kernel_size=3, padding=12, stride=1, dilation=12)
        self.drop = nn.Dropout(0.5)

        self.layer18 = BasicBlock(in_channel=1024, out_channel=1024, kernel_size=1, stride=1 , padding = 0)
        self.dropout = nn.Dropout(0.5)

        self.Conv2d = nn.Conv2d(kernel_size=1 , stride=1 , in_channels=1024 , out_channels=num_classes)
        

    def forward(self , x ):
        b , c, h , w = x.shape
        P1 = x.copy()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.max1(x)
        P2 = x.copy()

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.max2(x)
        P3 = x.copy()

        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.max4(x)
        P4 = x.copy()

        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.max5(x)
        P5 = x.copy()

        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.max6(x)

        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.max7(x)
        x = self.avg1(x)

        x = self.layer17(x)
        x = self.drop(x)

        x = self.layer18(x)
        x = self.dropout(x)

        logits = self.Conv2d(x)  # [B, C, h, w]
        P_final = logits.copy()
        out = F.interpolate(logits, size=(h , w), mode="bilinear", align_corners=False) 
        return out



class BackBone_VGG_16_MSC (nn.Module):
    def __init__(self , num_classes = 10):
        super(BackBone_VGG_16_MSC, self).__init__()
        self.numclasses = num_classes

        self.layer1 = BasicBlock(in_channel = 3 , out_channel= 64 , kernel_size= 3 , padding = 1 , stride = 1)
        self.layer2 = BasicBlock(in_channel = 64 , out_channel= 64 , kernel_size= 3 , padding = 1 , stride = 1)
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer3 = BasicBlock(in_channel=64, out_channel=128, kernel_size=3, padding=1, stride=1)
        self.layer4 = BasicBlock(in_channel=128, out_channel=128, kernel_size=3, padding=1, stride=1)
        self.max2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer5 = BasicBlock(in_channel=128, out_channel=256, kernel_size=3, padding=1, stride=1)
        self.layer6 = BasicBlock(in_channel=256, out_channel=256, kernel_size=3, padding=1, stride=1)
        self.layer7 = BasicBlock(in_channel=256, out_channel=256, kernel_size=3, padding=1, stride=1)
        self.max4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer8 = BasicBlock(in_channel=256, out_channel=512, kernel_size=3, padding=1, stride=1)
        self.layer9 = BasicBlock(in_channel=512, out_channel=512, kernel_size=3, padding=1, stride=1)
        self.layer10 = BasicBlock(in_channel=512, out_channel=512, kernel_size=3, padding=1, stride=1)
        self.max5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer11 = BasicBlock(in_channel=512, out_channel=512, kernel_size=3, padding=2, stride=1 ,dilation=2)
        self.layer12 = BasicBlock(in_channel=512, out_channel=512, kernel_size=3, padding=2, stride=1,dilation=2)
        self.layer13 = BasicBlock(in_channel=512, out_channel=512, kernel_size=3, padding=2, stride=1,dilation=2)
        self.max6 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)


        self.layer14 = BasicBlock(in_channel=512, out_channel=512, kernel_size=3, padding=2, stride=1, dilation=2)
        self.layer15 = BasicBlock(in_channel=512, out_channel=512, kernel_size=3, padding=2, stride=1, dilation=2)
        self.layer16 = BasicBlock(in_channel=512, out_channel=512, kernel_size=3, padding=2, stride=1, dilation=2)
        self.max7 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avg1 = nn.AvgPool2d(kernel_size=3 , stride=1 , padding=1)

        self.layer17 = BasicBlock(in_channel=512, out_channel=1024, kernel_size=3, padding=12, stride=1, dilation=12)
        self.drop = nn.Dropout(0.5)

        self.layer18 = BasicBlock(in_channel=1024, out_channel=1024, kernel_size=1, stride=1 , padding = 0)
        self.dropout = nn.Dropout(0.5)

        self.Conv2d = nn.Conv2d(kernel_size=1 , stride=1 , in_channels=1024 , out_channels=num_classes)

        self.conv_1 = nn.Conv2d(kernel_size=1 , stride= 1 , in_channels=1024 , out_channels=num_classes)
    
    def _make_msc_head(self , in_channel , stride):
        return nn.Sequential(
            BasicBlock(in_channel=in_channel, out_channel=128, kernel_size=3, padding=1, stride=stride),
            nn.Dropout(0.5),
            BasicBlock(in_channel=128, out_channel=128, kernel_size=1, padding=0, stride=1),
            nn.Dropout(0.5),
            nn.Conv2d(128, self.numclasses, kernel_size=1, stride=1, padding=0)
        )
    
    def _P1(self, x): return self.p1_head(x)
    def _P2(self, x): return self.p2_head(x)
    def _P3(self, x): return self.p3_head(x)
    def _P4(self, x): return self.p4_head(x)
    def _P5(self, x): return self.p5_head(x)


    def forward(self , x ):
        b , c, h , w = x.shape
        P1 = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.max1(x)
        P2 = x.copy()

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.max2(x)
        P3 = x

        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.max4(x)
        P4 = x

        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.max5(x)
        P5 = x

        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.max6(x)

        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.max7(x)
        x = self.avg1(x)

        x = self.layer17(x)
        x = self.drop(x)

        x = self.layer18(x)
        x = self.dropout(x)

        logits = self.Conv2d(x)  # [B, C, h, w]
        P_final = logits
        
        p1 = self._P1(P1)
        p2 = self._P2(P2)
        p3 = self._P3(P3)
        p4 = self._P4(P4)
        p5 = self._P5(P5)
        target_size = P_final.shape[-2:]

        # 对整齐大小尺寸等
        p1 = F.interpolate(p1, size=target_size, mode="bilinear", align_corners=False)
        p2 = F.interpolate(p2, size=target_size, mode="bilinear", align_corners=False)
        p3 = F.interpolate(p3, size=target_size, mode="bilinear", align_corners=False)
        p4 = F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False)
        p5 = F.interpolate(p5, size=target_size, mode="bilinear", align_corners=False)


        logits = P_final + p1 + p2 + p3 + p4 + p5 # adding

        # logits = torch.cat([P_final, p1, p2, p3, p4, p5], dim=1) # concat
        # logits = self.conv_1(logits)

        out = F.interpolate(logits, size=(h , w), mode="bilinear", align_corners=False) 

        return out
