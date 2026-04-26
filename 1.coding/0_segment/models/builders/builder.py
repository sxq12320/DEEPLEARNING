import torch
import torch.nn as nn
from ..registries.registry import BLOCK_REGISTRY

def make_layers(cfg):
    '''
    根据配置列表构建网络层序列。

    Args:
        cfg (list): 网络结构配置列表, 每个元素描述一个模块及其参数。

    Returns:
        nn.Sequential: 按配置构建得到的层序列。

    Raises:
        ValueError: 当配置项格式错误或 block 类型未注册时抛出。
    '''
    layers = []
    for idx, item in enumerate(cfg):
        if not isinstance(item, (list, tuple)) or len(item) == 0:
            raise ValueError(f"cfg[{idx}] must be a non-empty list/tuple, got: {item}")

        block_type = str(item[0]).strip().lower()
        repeat = int(item[-1]) if len(item) > 1 and isinstance(item[-1], int) else 1
        if repeat < 1:
            raise ValueError(f"cfg[{idx}] repeat must be >= 1, got: {repeat}")

        builder = BLOCK_REGISTRY.get(block_type)
        if builder is None:
            raise ValueError(
                f"Unsupported block type at cfg[{idx}]: {item[0]}. "
                f"Supported: {list(BLOCK_REGISTRY.keys())}"
            )

        # 第一次构建
        block = builder(item)
        layers.append(block)

        # 重复构建
        for _ in range(1, repeat):
            # 对于重复层，你可能需要调整参数（比如 stride 置 1）
            # 可以在这里处理，或者在 builder 内部根据 repeat 判断
            # 简单起见，这里直接复用同一个 builder，你可以根据需求修改
            layers.append(builder(item))

    return nn.Sequential(*layers)







# def make_layers(cfg):

#     '''
#     简化模块构建的繁琐方式，具体的模块包括:
    
#     "maxpool":kernel_size , stride_size , padding_size , dilation_size
#     "adaptive_avg_pool":output_size
#     "conv" :in_channels , out_channels , kernel_size , stride_size , padding_size , dilation_size , groups_size , bias
#     "basic_conv_block" :in_channels , out_channels , kernel_size , stride_size , padding_size , dilation_size , groups_size , activation
#     "conv_block_nonb" :in_channels , out_channels , kernel_size , stride_size , padding_size , dilation_size , groups_size , activation
#     "depthwise_conv":in_channels , kernel_size , stride_size , padding_size , dilation_size 
#     "pointwise_conv":in_channels , out_channels , stride_size 
#     "depthwise_separable_conv":in_channels , out_channels , kernel_size , padding_size , stride_size_D , stride_size_P , dilation_size_D , activation
#     "resnet_block_34":in_channels , out_channels , stride_size , activation_1  , activation_2
#     "resnet_block_50":in_channels , out_channels , stride_size , activation_1 , activation_2 , activation_3 , expansion_size
#     "cbam_channel_attention":in_channels , reduction_ratio , activation
#     "cbam_spatial_attention":kernel_size
#     "cbam":in_channels , reduction_ratio , activation , kernel_size
#     "flatten":None
#     "linear":in_features , out_features , bias
    
#     Args:
#         各种类型的参数，需要后续自行进行配置即可
    
#     Returns:
#         nn.Sequential: 构建好的网络层
  
#     '''
#     layers = []     
#     for idx, item in enumerate(cfg):
#         if not isinstance(item, (list, tuple)) or len(item) == 0:
#             raise ValueError(f"cfg[{idx}] must be a non-empty list/tuple, got: {item}")

#         block_type = str(item[0]).strip().lower()
#         repeat = int(item[-1]) if len(item) > 1 and isinstance(item[-1], int) else 1
#         if repeat < 1:
#             raise ValueError(f"cfg[{idx}] repeat must be >= 1, got: {repeat}")
        

#         if block_type == 'conv':
#             '''
#             构建不存在激活函数和批量归一化的卷积块
#             '''
#             if len(item) != 10:
#                 raise ValueError(f"cfg[{idx}] conv 期望的参数个数为10, 现在输入的参数是 {len(item)}: {item}")
#             bloack_name , in_channels , out_channels , kernel_size , stride_size , padding_size , dilation_size , groups_size , bias , repeat = item
#             layers.append(
#                 Conv(
#                     in_ch=in_channels,
#                     out_ch=out_channels,
#                     k=kernel_size,
#                     s=stride_size,
#                     p=padding_size,
#                     d=dilation_size,
#                     g=groups_size,
#                     b=bias,
#                 )
#             )
#             for _ in range(1 , repeat):
#                 layers.append(
#                     Conv(
#                         in_ch=out_channels,
#                         out_ch=out_channels,
#                         k=kernel_size,
#                         s=1,
#                         p=padding_size,
#                         d=dilation_size,
#                         g=groups_size,
#                         b=bias
#                     )
#                 )
        
#         elif block_type == 'maxpool':
#             '''
#             构建最大池化函数
#             '''    
#             if len(item) != 5:
#                 raise ValueError(f"cfg[{idx}] maxpool 期望的参数个数为5, 现在输入的参数是 {len(item)}: {item}")
#             block_name , kernel_size , stride_size , padding_size , dilation_size = item
#             layers.append(
#                 MaxPool(
#                     k=kernel_size,
#                     s=stride_size,
#                     p=padding_size,
#                     d=dilation_size
#                 )
#             )
        
#         elif block_type == 'adaptive_avg_pool':
#             '''
#             构建自适应平均池化函数
#             '''
#             if len(item) != 3:
#                 raise ValueError(f"cfg[{idx}] adaptive_avg_pool 期望的参数个数为3, 现在输入的参数是 {len(item)}: {item}")
#             block_name , output_size , repeat = item
#             layers.append(
#                 AdaptiveAvgPool(
#                     output_size=output_size
#                 )
#             )
#             for _ in range(1 , repeat):
#                 layers.append(
#                     AdaptiveAvgPool(
#                         output_size=output_size
#                     )
#                 )

#         elif block_type == 'basic_conv_block':
#             '''
#             构建基本卷积块 
#             '''
#             if len(item) != 10:
#                 raise ValueError(f"cfg[{idx}] basic_conv_block 期望的参数个数为10, 现在输入的参数是 {len(item)}: {item}")
#             block_name , in_channels , out_channels , kernel_size , stride_size , padding_size , dilation_size , groups_size , activation , repeat = item
#             layers.append(
#                 Basic_Conv_Block(
#                 in_ch=in_channels,
#                 out_ch=out_channels,
#                 k=kernel_size,
#                 s=stride_size,
#                 p=padding_size,
#                 d=dilation_size,
#                 g=groups_size,
#                 activation=activation
#                 )
#             )
#             for _ in range(1 , repeat):
#                 layers.append(Basic_Conv_Block(
#                     in_ch=out_channels,
#                     out_ch=out_channels,
#                     k=kernel_size,
#                     s=1, # 设置为1防止出现奇奇怪怪的问题
#                     p=padding_size,
#                     d=dilation_size,
#                     g=groups_size,
#                     activation=activation
#                 ))

#         elif block_type == 'conv_basic_nonb':
#             '''
#             构建没有批归一化的基本的卷积块
#             '''
#             if len(item) != 10:
#                 raise ValueError(f"cfg[{idx}] conv_basic_nonb 期望的参数个数为10, 现在输入的参数是{len(item)}: {item}")
#             block_name , in_channels , out_channels , kernel_size , stride_size , padding_size , dilation_size , groups_size , activation ,repeat = item
#             layers.append(
#                 Conv_Block_NONB(
#                     in_ch=in_channels,
#                     out_ch = out_channels,
#                     k=kernel_size,
#                     s=stride_size,
#                     p=padding_size,
#                     d=dilation_size,
#                     g=groups_size,
#                     activation=activation
#                 )
#             )
#             for _ in range(1 , repeat):
#                 layers.append(
#                     Conv_Block_NONB(
#                         in_ch=out_channels,
#                         out_ch=out_channels,
#                         k=kernel_size,
#                         s=1,
#                         p=padding_size,
#                         d=dilation_size,
#                         g=groups_size,
#                         activation=activation
#                     )
#                 )

#         elif block_type == 'depthwise_conv':
#             '''
#             构建基本的深度分离卷积模块
#             '''
#             if len(item) != 7:
#                 raise ValueError(f"cfg[{idx}] depthwise_conv 期望的参数个数为7, 现在输入的参数是{len(item)}: {item}")
#             block_name , in_channels , kernel_size , stride_size , padding_size , dilation_size , repeat = item
#             layers.append(
#                 DepthWise_Conv(
#                     in_ch= in_channels,
#                     k = kernel_size,
#                     s = stride_size,
#                     p = padding_size,
#                     d = dilation_size,
#                 )
#             )
#             for _ in range(1 , repeat):
#                 layers.append(
#                     DepthWise_Conv(
#                     in_ch= in_channels,
#                     k = 1,
#                     s = stride_size,
#                     p = padding_size,
#                     d = dilation_size,
#                     )
#                 )
        
#         elif block_type == "pointwise_conv":
#             '''
#             逐点卷积的基本模块
#             '''
#             if len(item) != 5:
#                 raise ValueError(f"cfg[{idx}] pointwise_conv 期望的参数个数为5, 现在输入的参数是{len(item)}: {item}")
#             block_name , in_channels , out_channels , stride_size , repeat = item
#             layers.append(
#                 PointWise_Conv(
#                     in_ch=in_channels , 
#                     out_ch= out_channels,
#                     s = stride_size,
#                 )
#             )
#             for _ in range(1 , repeat):
#                 layers.append(
#                     PointWise_Conv(
#                     in_ch=in_channels , 
#                     out_ch= out_channels,
#                     s = 1,
#                     )
#                 )
        
#         elif block_type == 'depthwise_separable_conv':
#             '''
#             深度可分离卷积的基本模块
#             '''
#             if len(item) != 10:
#                 raise ValueError(f"cfg[{idx}] depthwise_separable_conv 期望的参数个数为10, 现在输入的参数是{len(item)}: {item}")
#             block_name , in_channels , out_channels , kernel_size , padding_size , stride_size_D , stride_size_P , dilation_size_D , activation , repeat = item
#             layers.append(
#                 DepthWiseSeparable_Conv(
#                     in_ch=in_channels,
#                     out_ch=out_channels,
#                     k=kernel_size,
#                     p=padding_size,
#                     s_D=stride_size_D,
#                     s_P=stride_size_P,
#                     d_D=dilation_size_D,
#                     activation=activation
#                 )
#             )
#             for _ in range(1 , repeat):
#                 layers.append(
#                     DepthWiseSeparable_Conv(
#                         in_ch=in_channels,
#                         out_ch=out_channels,
#                         k=kernel_size,
#                         p=padding_size,
#                         s_D=stride_size_D,
#                         s_P=stride_size_P,
#                         d_D=dilation_size_D,
#                         activation=activation
#                     )
#                 )

#         elif block_type == "resnet_block_34":
#             '''
#             ResNet-34 基本模块
#             '''
#             if len(item) != 7:
#                 raise ValueError(f"cfg[{idx}] resnet_block_34 期望的参数个数为7, 现在输入的参数是{len(item)}: {item}")
#             block_name , in_channels , out_channels , stride_size , activation_1  , activation_2 , repeat = item
#             layers.append(
#                 ResNetBlock_34(
#                     in_ch = in_channels,
#                     out_ch=out_channels,
#                     s = stride_size,
#                     activation_1 = activation_1,
#                     activation_2 = activation_2
#                 )
#             )
#             for _ in range(1 , repeat):
#                 layers.append(
#                     ResNetBlock_34(
#                     in_ch = out_channels,
#                     out_ch=out_channels,
#                     s = 1,
#                     activation_1 = activation_1,
#                     activation_2 = activation_2
#                     )
#                 )
        
#         elif block_type == "resnet_block_50":
#             '''
#             ResNet-50 基本模块
#             '''
#             if len(item) != 9:
#                 raise ValueError(f"cfg[{idx}] resnet_block_50 期望的参数个数为9, 现在输入的参数是{len(item)}: {item}")
#             block_name , in_channels , out_channels , stride_size , activation_1  , activation_2 , activation_3 , expansion_size , repeat = item
#             layers.append(
#                 ResNetBlock_50(
#                     in_ch = in_channels,
#                     out_ch=out_channels,
#                     s = stride_size,
#                     activation_1 = activation_1,
#                     activation_2 = activation_2,
#                     activation_3 = activation_3,
#                     expansion_size = expansion_size
#                 )
#             )
#             for _ in range(1 , repeat):
#                 layers.append(
#                     ResNetBlock_50(
#                     in_ch = out_channels,
#                     out_ch=out_channels,
#                     s = 1,
#                     activation_1 = activation_1,
#                     activation_2 = activation_2,
#                     activation_3 = activation_3,
#                     expansion_size = expansion_size
#                     )
#                 )

#         elif block_type == "cbam_channel_attention":
#             '''
#             CBAM通道注意力机制
#             '''
#             if len(item) != 5:
#                 raise ValueError(f"cfg[{idx}] cbam_channel_attention 期望的参数个数为4, 现在输入的参数是{len(item)}: {item}")
#             block_name , in_channels , reductiaon_ratio , activation ,  repeat = item
#             layers.append(
#                 CBAM_Channel_Attention(
#                     in_ch=in_channels,
#                     reduction_ratio=reductiaon_ratio,
#                     activation= activation
#                 )
#             )
#             for _ in range(1 , repeat):
#                 layers.append(
#                     CBAM_Channel_Attention(
#                         in_ch=in_channels,
#                         reduction_ratio=reductiaon_ratio,
#                         activation= activation
#                     )
#                 )

#         elif block_type == "cbam_spatial_attention":
#             '''
#             CBAM空间注意力机制
#             '''
#             if len(item) != 3:
#                 raise ValueError(f"cfg[{idx}] cbam_spatial_attention 期望的参数个数为4, 现在输入的参数是{len(item)}: {item}")
#             block_name  , kernel_size ,  repeat = item
#             layers.append(
#                 CBAM_Spatial_Attention(
#                     k=kernel_size
#                 )
#             )
#             for _ in range(1 , repeat):
#                 layers.append(
#                     CBAM_Spatial_Attention(
#                         k=kernel_size
#                     )
#                 )

#         elif block_type == "cbam":
#             '''
#             CBAM注意力机制
#             '''
#             if len(item) != 6:
#                 raise ValueError(f"cfg[{idx}] cbam 期望的参数个数为6, 现在输入的参数是{len(item)}: {item}")
#             block_name , in_channels , reduction_ratio , activation , kernel_size ,  repeat = item
#             layers.append(
#                 CBAM(
#                     in_ch=in_channels,
#                     reduction_ratio=reduction_ratio,
#                     activation=activation,
#                     k=kernel_size
#                 )
#             )
#             for _ in range(1 , repeat):
#                 layers.append(
#                     CBAM(
#                         in_ch=in_channels,
#                         reduction_ratio=reduction_ratio,
#                         activation=activation,
#                         k=kernel_size
#                     )
#                 )

#         elif block_type == "flatten":
#             '''
#             Faltten展平函数
#             '''
#             if len(item) != 2:
#                 raise ValueError(f"cfg[{idx}] flatten 期望的参数个数为2, 现在输入的参数是{len(item)}: {item}")
#             block_name ,  repeat = item
#             layers.append(
#                 Flatten()
#             )
#             for _ in range(1 , repeat):
#                 layers.append(
#                     Flatten()
#                 )

#         elif block_type == "linear":
#             '''
#             线性层
#             '''
#             if len(item) != 5:
#                 raise ValueError(f"cfg[{idx}] linear 期望的参数个数为5, 现在输入的参数是{len(item)}: {item}")
#             block_name , in_features , out_features , bias , repeat = item
#             layers.append(
#                 Linear(
#                     in_feature=in_features,
#                     out_feature=out_features,
#                     bias=bias
#                 )
#             )
#             for _ in range(1 , repeat):
#                 layers.append(
#                     Linear(
#                         in_feature=in_features,
#                         out_feature=out_features,
#                         bias=bias
#                     )
#                 )

#         else:
#             raise ValueError(
#                 f"Unsupported block type at cfg[{idx}]: {item[0]}. "
#                 "Supported: basic_conv_block, conv_block_nonb, depthwise_conv, pointwise_conv, "
#                 "depthwise_separable_conv, res34, cbam_channel_attention, cbam_spatial_attention, cbam"
#             )

#     return nn.Sequential(*layers)

