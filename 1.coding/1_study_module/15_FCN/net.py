import torch
import torch.nn as nn


def conv_block(in_channels, out_channels, num_convs):
    '''
        VGG网络的基本块儿
    '''
    layers = []
    for _ in range(num_convs):
        layers += [
            nn.Conv2d(in_channels , out_channels , kernel_size=3 , padding=1 , bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2 , stride = 2))
    return nn.Sequential(*layers)

class VGG_16(nn.Module):
    '''
    VGGnet的整体架构
    '''
    def __init__(self):
        super().__init__()
        self.pool1 = conv_block(3, 64, 2)
        self.pool2 = conv_block(64, 128, 2)
        self.pool3 = conv_block(128, 256, 3)
        self.pool4 = conv_block(256, 512, 3)
        self.pool5 = conv_block(512, 512, 3)
        self._init_weights()

    def forward(self, x):
        x = self.pool1(x)
        x = self.pool2(x)
        p3 = self.pool3(x)
        p4 = self.pool4(p3)
        p5 = self.pool5(p4)
        return p3, p4, p5
    
    def _init_weights(self):
        for m in self.modules():# 遍历全部的目标子模块
            if isinstance(m, nn.Conv2d): # 判定该层是否为卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')#凯明初始化
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def _make_fc_conv():
    """VGG FC6+FC7 卷积化，kernel=1 适配任意输入尺寸"""
    return nn.Sequential(
        nn.Conv2d(512 , 4096 , kernel_size=1 , bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout2d(0.5),
        nn.Conv2d(4096, 4096, kernel_size=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout2d(0.5)
    )

def _init_fc_conv(module):
    """初始化FC卷积层"""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                    nonlinearity='relu')
            nn.init.constant_(m.bias, 0)


def _bilinear_init(layer: nn.ConvTranspose2d):
    """双线性插值初始化转置卷积（论文 3.3 节）"""
    w = layer.weight.data
    f = w.shape[2]
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(f):
        for j in range(f):
            val = (1 - abs(i / f - c)) * (1 - abs(j / f - c))
            w[:, :, i, j] = val
 
def _crop(x: torch.Tensor, target_size):
    """中心裁剪，消除转置卷积输出与目标尺寸的偏差"""
    h, w   = target_size
    oh, ow = x.shape[2], x.shape[3]
    dh = (oh - h) // 2
    dw = (ow - w) // 2
    return x[:, :, dh:dh + h, dw:dw + w]



class FCN32s(nn.Module):
    def __init__(self , num_classes: int):
        super(FCN32s, self).__init__()
        self.backbone = VGG_16()
        self.fc_conv = _make_fc_conv()
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1) # 分类头
        self.upscore32= nn.ConvTranspose2d(# 转置卷积
            num_classes, num_classes , 
            kernel_size=64 , 
            stride=32 , 
            padding=16 , 
            bias=False
        )
        self._init_weights()

    def forward(self , x):
        input_size    = x.shape[2:]
        _, _, p5      = self.backbone(x)
        x = self.fc_conv(p5)
        x = self.score_fr(x)
        x = self.upscore32(x)
        return _crop(x, input_size)


    def _init_weights(self):
        _init_fc_conv(self.fc_conv)
        nn.init.kaiming_normal_(self.score_fr.weight)
        nn.init.constant_(self.score_fr.bias, 0)
        _bilinear_init(self.upscore32)



def build_model(model_type: str, num_classes: int) -> nn.Module:
    """model_type: 'fcn32s' | 'fcn16s' | 'fcn8s'"""
    model_map = {
        'fcn32s': FCN32s,
        'fcn16s': FCN16s,
        'fcn8s':  FCN8s,
    }
    assert model_type in model_map, f"未知模型类型: {model_type}"
    return model_map[model_type](num_classes)
 
 
if __name__ == '__main__':
    x = torch.randn(2, 3, 512, 512)
    for name in ['fcn32s', 'fcn16s', 'fcn8s']:
        model = build_model(name, num_classes=20)
        out   = model(x)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{name}: input {list(x.shape)} → output {list(out.shape)}  参数量: {params:.1f}M")