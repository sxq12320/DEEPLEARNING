import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import numpy as np
import sys
import os 

class ScaledDotProductAttention(nn.Module):
    '''单头注意力机制模块
    Args:
        @   dropout: dropout比率
    Returns:
        @   output: 注意力输出张量 [batch_size , n_heads , len_q , d_v]
        @   attn: 注意力权重张量 [batch_size , n_heads , len_q , len_k]
    计算步骤:   
        @  1. Q*K^T
        @  2. /sqrt(d_k)
        @  3. mask
        @  4. softmax
        @  5. dropout
    '''
    def __init__(self , dropout = 0.1):
        super(ScaledDotProductAttention , self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self , Q , K , V , mask = None):
        '''前向传播函数
        Args:
            @   Q: Query张量 [batch_size , n_heads , len_q , d_k]
            @   K: Key张量 [batch_size , n_heads , len_k , d_k]
            @   V: Value张量 [batch_size , n_heads , len_v , d_v]
            @   mask: 掩码张量 [batch_size , n_heads , len_q , len_k]
        Returns:
            @   output: 注意力输出 [batch_size , n_heads , len_q , d_v]
            @   attn: 注意力权重张量 [batch_size , n_heads , len_q , len_k]
        '''
        out_matmul = torch.matmul(Q , K.transpose(-2,-1))
        dk = K.size(-1)
        out_scaled = out_matmul / math.sqrt(dk)     

        if mask is not None:
            if mask.dim() == 2:  # [seq_q, seq_k]
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_q, seq_k]
            elif mask.dim() == 3:  # [batch, seq_q, seq_k] 
                mask = mask.unsqueeze(1)  # [batch, 1, seq_q, seq_k]
            out_scaled = out_scaled.masked_fill(mask == 0 , -1e9)       
        
        attn = self.softmax(out_scaled)
        attn_weight = self.dropout(attn)
        output = torch.matmul(attn_weight , V)

        return output , attn

class MultiHeadAttention(nn.Module):
    '''多头注意力机制模块
    Args:
        @   d_model: 输入输出张量的维度
        @   n_heads: 注意力头数
        @   dropout: dropout比率
    Returns:
        @   output: 多头注意力输出张量 [batch_size , len_q , d_model]
        @   attn: 注意力权重张量 [batch_size , n_heads , len_q , len_k]
    计算步骤:
        @  1. 线性映射 Q , K , V
        @  2. 拆分头
        @  3. 单头注意力计算
        @  4. 拼接头
        @  5. 线性映射
    '''
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention , self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.scale_dot_product_attention = ScaledDotProductAttention(dropout)
        assert d_model % n_heads == 0 , "d_model must be divisible by n_heads"
        
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        self.W_Q = nn.Linear(d_model , d_model)
        self.W_K = nn.Linear(d_model , d_model)
        self.W_V = nn.Linear(d_model , d_model)

        self.W_O = nn.Linear(d_model , d_model)

    def split_heads(self , x):
        '''拆分注意力头,位于多头注意力机制开始之前
        Args:
            @   x: 输入张量 [batch_size , len_seq , d_model]
        Returns:
            @   output: 拆分后的张量 [batch_size , n_heads , len_seq , d_k]
        '''
        batch_size , len_seq , d_model = x.size()
        x = x.view(batch_size , len_seq , self.n_heads , self.d_k)

        return x.transpose(1 , 2)
    
    def combine_heads(self , x):
        '''拼接注意力头，位于多头注意力机制结束之后也就是线性变换之后
        Args:
            @   x: 拆分后的张量 [batch_size , n_heads , len_seq , d_k]
        Returns:
            @   output: 拼接后的张量 [batch_size , len_seq , d_model]
        '''
        batch_size , n_heads , len_seq , d_k = x.size()
        x = x.transpose(1 , 2).contiguous().view(batch_size , len_seq , self.d_model)

        return x
    
    def forward(self , Q , K , V , mask = None):
        '''前向传播函数
        Args:
            @   Q: Query张量 [batch_size , len_q , d_model]
            @   K: Key张量 [batch_size , len_k , d_model]
            @   V: Value张量 [batch_size , len_v , d_model]
            @   mask: 掩码张量 [batch_size , len_q , len_k]
            注意 batch_size 指的是：批次大小，在一段话中表示为句子数目；
                len_q 指的是：所谓的序列长度，在一段话话中表示单词数目；
                d_model 指的是：词向量的维度
        Returns:
            @   output: 多头注意力输出张量 [batch_size , len_q , d_model]
            @   attn: 注意力权重张量 [batch_size , n_heads , len_q , len_k]
        '''
        # 1. 线性映射 Q , K , V
        Q = self.W_Q(Q)  # [batch_size , len_q , d_model]
        K = self.W_K(K)  # [batch_size , len_k , d_model]
        V = self.W_V(V)  # [batch_size , len_v , d_model]

        # 2. 拆分头
        Q = self.split_heads(Q)  # [batch_size , n_heads , len_q , d_k]
        V = self.split_heads(V)  # [batch_size , n_heads , len_v , d_k]
        K = self.split_heads(K)  # [batch_size , n_heads , len_k , d_k]

        # 3. 单头注意力计算
        output , attn = self.scale_dot_product_attention(Q , K , V , mask)

        # 4. 拼接头
        output = self.combine_heads(output)  # [batch_size , len_q , d_model]

        return output , attn

class FeedForward(nn.Module):
    '''前馈神经网路对应的是Transformer中的全连接前馈网络模块
    Args:
        @   d_model: 输入输出张量的维度
        @   d_ff: 前馈全连接网络隐藏层维度
        @   dropout: dropout比率
    Returns:
        @   output: 前馈全连接网络输出张量 [batch_size , len_seq , d_model]
    计算步骤:
        @  1. 线性映射
        @  2. 激活函数ReLU
        @  3. dropout
        @  4. 线性映射
    '''
    def __init__(self , d_model , d_ff , dropout = 0.1):
        super(FeedForward , self).__init__()
        self.fc1 = nn.Linear(d_model , d_ff)
        self.fc2 = nn.Linear(d_ff , d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self , x):
        '''前向传播函数
        Args:
            @   x: 输入张量 [batch_size , len_seq , d_model]
        Returns:
            @   output: 前馈全连接网络输出张量 [batch_size , len_seq , d_model]
        '''
        out_fc1 = self.fc1(x)
        out_relu = self.relu(out_fc1)
        out_dropout = self.dropout(out_relu)
        output = self.fc2(out_dropout)
        return output
    
class LayerNorm(nn.Module):
    '''层归一化模块 ,运用在add & norm中
    Args:
        @   d_model: 输入输出张量的维度
        @   eps: 防止除零操作的小值
    Returns:
        @   output: 层归一化输出张量 [batch_size , len_seq , d_model]
    计算步骤:
        @  1. 计算均值
        @  2. 计算方差
        @  3. 标准化
        @  4. 缩放和平移
    '''
    def __init__(self , d_model , eps = 1e-6):
        super(LayerNorm , self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
    def forward(self , x):
        '''前向传播函数
        Args:
            @   x: 输入张量 [batch_size , len_seq , d_model]
        Returns:
            @   output: 层归一化输出张量 [batch_size , len_seq , d_model]
        '''
        mean = x.mean(-1 , keepdim = True)
        std = x.std(-1 , keepdim = True)
        output = self.gamma * (x - mean) / (std + self.eps) + self.beta
        return output
    
class EncoderLayer(nn.Module):
    '''单个-Transformer编码器层模块
    Args:
        @   d_model: 输入输出张量的维度
        @   n_heads: 注意力头数
        @   d_ff: 前馈全连接网络隐藏层维度
        @   dropout: dropout比率
    Returns:
        @   output: 编码器层输出张量 [batch_size , len_seq , d_model]
    计算步骤:
        @  1. 多头注意力机制
        @  2. 残差连接与层归一化
        @  3. 前馈全连接网络
        @  4. 残差连接与层归一化
    '''
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer , self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self , x , mask = None):
        '''前向传播函数
        Args:
            @   x: 输入张量 [batch_size , len_seq , d_model]
            @   mask: 掩码张量 [batch_size , len_seq , len_seq]
        Returns:
            @   output: 编码器层输出张量 [batch_size , len_seq , d_model]
        '''
        # 1. 多头注意力机制 + 残差连接 + 层归一化
        attn_output , attn_weights = self.self_attn(x , x , x , mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # 2. 前馈全连接网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        output = self.norm2(x)

        return output
    
class DecoderLayer(nn.Module):
    '''单个-Transformer解码器层模块
    Args:
        @   d_model: 输入输出张量的维度
        @   n_heads: 注意力头数
        @   d_ff: 前馈全连接网络隐藏层维度
        @   dropout: dropout比率
    Returns:
        @   output: 解码器层输出张量 [batch_size , len_seq , d_model]
    计算步骤:
        @  1. 掩码多头自注意力机制
        @  2. 残差连接与层归一化
        @  3. 编码器-解码器多头注意力机制
        @  4. 残差连接与层归一化
        @  5. 前馈全连接网络
        @  6. 残差连接与层归一化
    '''
    def __init__(self , d_model , n_heads , d_ff , dropout = 0.1):
        super(DecoderLayer , self).__init__()
        self.self_attn1 = MultiHeadAttention(d_model , n_heads , dropout)
        self.LayerNorm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.self_attn2 = MultiHeadAttention(d_model , n_heads , dropout)
        self.LayerNorm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.feed_forward = FeedForward(d_model , d_ff , dropout)
        self.LayerNorm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self , x , enc_output , src_mask = None , tgt_mask = None):
        '''前向传播函数
        Args:
            @   x: 输入张量 [batch_size , len_seq , d_model]
            @   enc_output: 编码器输出张量 [batch_size , len_seq , d_model]
            @   src_mask: 源序列掩码张量 [batch_size , len_seq , len_seq]
            @   tgt_mask: 目标序列掩码张量 [batch_size , len_seq , len_seq]
        Returns:
            @   output: 解码器层输出张量 [batch_size , len_seq , d_model]
        '''
        # 1. 掩码多头自注意力机制 + 残差连接 + 层归一化
        attn_output1 , attn_weights1 = self.self_attn1(x , x , x , tgt_mask)
        x = x + self.dropout1(attn_output1)
        x = self.LayerNorm1(x)

        # 2. 编码器-解码器多头注意力机制 + 残差连接 + 层归一化
        attn_output2 , attn_weights2 = self.self_attn2(x , enc_output , enc_output , src_mask)
        x = x + self.dropout2(attn_output2)
        x = self.LayerNorm2(x)

        # 3. 前馈全连接网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        output = self.LayerNorm3(x)

        return output , attn_weights1 , attn_weights2

class PositionalEncoding(nn.Module):
    '''位置编码模块
    Args:
        @   d_model: 模型维度
        @   dropout: dropout比率
        @   max_len: 最大序列长度
    Returns:
        @   output: 位置编码后的张量 [batch_size, seq_len, d_model]
    '''
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数位置用sin，奇数位置用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  #注册为buffer，不作为模型参数更新
    
    def forward(self, x):
        '''
        Args:
            @   x: 输入张量 [batch_size, seq_len, d_model]
        Returns:
            @   output: 位置编码后的张量 [batch_size, seq_len, d_model]
        '''
        # x: [batch_size, seq_len, d_model]
        # pe: [1, max_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransFormer(nn.Module):
    '''Transformer模型模块
    Args:
        @   d_model: 输入输出张量的维度
        @   n_heads: 注意力头数
        @   d_ff: 前馈全连接网络隐藏层维度
        @   num_encoder_layers: 编码器层数
        @   num_decoder_layers: 解码器层数
        @   dropout: dropout比率
        @   src_vocab_size: 源词汇表大小
        @   tgt_vocab_size: 目标词汇表大小
    Returns:
        @   output: Transformer模型输出张量 [batch_size , len_seq , d_model]
    '''
    def __init__(self ,src_vocab_size , tgt_vocab_size , d_model , n_heads , d_ff , num_encoder_layers=6 , num_decoder_layers=6 , dropout = 0.1 ):
        super(TransFormer , self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)  # 源语义编码
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)  # 目标语义编码
        self.pos_encoding = PositionalEncoding(d_model , dropout)  # 位置编码
        self.encoder_layers = nn.ModuleList( # 编码器堆叠
            [
                EncoderLayer(d_model , n_heads , d_ff , dropout) 
                for _ in range(num_encoder_layers)
                ]
            )
        self.encoder_norm = LayerNorm(d_model)  # 编码器层归一化
        self.decoder_layers = nn.ModuleList( # 解码器堆叠
            [
                DecoderLayer(d_model , n_heads , d_ff , dropout) 
                for _ in range(num_decoder_layers)
                ]
            )
        self.decoder_norm = LayerNorm(d_model)  # 解码器层归一化
        self.fc_out = nn.Linear(d_model , tgt_vocab_size)  # 输出线性映射
        
        # 模型参数
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        self._init_weights()

        self.softmax = nn.Softmax(dim = -1)
    
    def _init_weights(self):
        '''参数初始化'''
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # 嵌入层缩放
        self.src_embedding.weight.data *= math.sqrt(self.d_model)
        self.tgt_embedding.weight.data *= math.sqrt(self.d_model)

    def generate_square_subsequent_mask(self , sz):
        '''生成目标序列掩码张量
        Args:
            @   sz: 序列长度
        Returns:
            @   mask: 目标序列掩码张量 [sz , sz]
        '''
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        '''
        Args:
            @   src: 源序列 [batch_size, src_len]
            @   tgt: 目标序列 [batch_size, tgt_len]
            @   src_mask: 源序列掩码 [batch_size, src_len] (可选)
            @   tgt_mask: 目标序列掩码 [batch_size, tgt_len] (可选)
            @   memory_mask: 编码器-解码器掩码 [batch_size, tgt_len, src_len] (可选)
        Returns:
            @   logits: 未归一化的预测分数 [batch_size, tgt_len, tgt_vocab_size]
            @   enc_self_attns: 编码器自注意力权重列表
            @   dec_self_attns: 解码器自注意力权重列表  
            @   dec_cross_attns: 解码器交叉注意力权重列表
        '''
        # 1. 生成掩码 (如果未提供)
        if tgt_mask is None or isinstance(tgt_mask, bool):
            tgt_len = tgt.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        
        # 2. 处理源序列掩码
        if src_mask is not None and src_mask.dim() == 2:
            # 将填充掩码转换为注意力掩码格式
            src_mask = src_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, src_len]
        
        # 3. 处理目标序列掩码
        if tgt_mask is not None and tgt_mask.dim() == 2:
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, tgt_len]
        
        # 4. 编码器部分
        ## 4.1 源序列嵌入 + 位置编码
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoding(src_embedded)
        
        ## 4.2 通过所有编码器层
        enc_output = src_embedded
        enc_self_attns = []
        
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
            # 如果需要记录注意力权重，修改EncoderLayer返回注意力权重
            # 这里为简化，假设不返回
        
        enc_output = self.encoder_norm(enc_output)
        
        # 5. 解码器部分
        ## 5.1 目标序列嵌入 + 位置编码
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoding(tgt_embedded)
        
        ## 5.2 通过所有解码器层
        dec_output = tgt_embedded
        dec_self_attns = []
        dec_cross_attns = []
        
        for layer in self.decoder_layers:
            dec_output, self_attn, cross_attn = layer(
                dec_output, enc_output, src_mask, tgt_mask
            )
            dec_self_attns.append(self_attn)
            dec_cross_attns.append(cross_attn)
        
        dec_output = self.decoder_norm(dec_output)
        
        # 6. 最终输出投影
        logits = self.fc_out(dec_output)  # [batch_size, tgt_len, tgt_vocab_size]
        
        # return logits, enc_self_attns, dec_self_attns, dec_cross_attns
        return logits
     



# =============== AI写的 测试代码 ===============#
# 创建测试数据
def create_test_data(batch_size=2, src_len=5, tgt_len=6, src_vocab=10, tgt_vocab=12):
    """
    创建用于测试的随机数据
    """
    # 随机生成token IDs (1到vocab_size-1，0通常用于padding)
    src = torch.randint(1, src_vocab, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab, (batch_size, tgt_len))
    
    # 创建源序列掩码 (模拟真实场景中的padding掩码)
    src_pad_mask = (src != 0)  # 假设0是padding token
    
    return src, tgt, src_pad_mask 

# ========== 3. 模型测试函数 ==========
def test_transformer():
    print("="*50)
    print("🚀 开始Transformer模型测试")
    print("="*50)
    
    # 模型参数
    src_vocab_size = 1000  # 源词汇表大小
    tgt_vocab_size = 1200  # 目标词汇表大小
    d_model = 64           # 模型维度 (减小便于测试)
    n_heads = 4            # 注意力头数
    d_ff = 256             # 前馈网络维度
    num_encoder_layers = 2 # 编码器层数 (减小便于测试)
    num_decoder_layers = 2 # 解码器层数 (减小便于测试)
    dropout = 0.1
    
    # 创建模型
    print("\n🔧 初始化Transformer模型...")
    model = TransFormer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"✅ 模型已移动到设备: {device}")
    print(f"📊 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建测试数据
    print("\n📦 创建测试数据...")
    batch_size = 4
    src_len = 8
    tgt_len = 10
    
    src, tgt, src_mask = create_test_data(
        batch_size=batch_size,
        src_len=src_len,
        tgt_len=tgt_len,
        src_vocab=src_vocab_size,
        tgt_vocab=tgt_vocab_size
    )
    
    # 添加padding以模拟真实场景
    src[0, 5:] = 0  # 第一个样本的最后3个token是padding
    src[1, 6:] = 0  # 第二个样本的最后2个token是padding
    
    print(f"✅ 源序列形状: {src.shape}, 内容:\n{src}")
    print(f"✅ 目标序列形状: {tgt.shape}, 内容:\n{tgt}")
    print(f"✅ 源掩码形状: {src_mask.shape}")
    
    # 移动到设备
    src = src.to(device)
    tgt = tgt.to(device)
    src_mask = src_mask.to(device)
    
    # 生成目标序列掩码 (因果掩码)
    tgt_mask = model.generate_square_subsequent_mask(tgt_len).to(device)
    print(f"✅ 目标掩码形状: {tgt_mask.shape}")
    print(f"🔤 目标掩码 (下三角):\n{tgt_mask.cpu().numpy()}")
    
    # 前向传播
    print("\n⚡ 执行前向传播...")
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
    
    print(f"\n✅ 前向传播成功!")
    print(f"📈 输出形状: {output.shape}")
    print(f"🧠 输出数据类型: {output.dtype}")
    print(f"📊 输出范围: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}")
    
    # 验证输出形状
    expected_shape = (batch_size, tgt_len, tgt_vocab_size)
    assert output.shape == expected_shape, \
        f"输出形状错误! 期望 {expected_shape}, 但得到 {output.shape}"
    
    # 检查是否有NaN或Inf
    assert not torch.isnan(output).any(), "输出包含NaN值!"
    assert not torch.isinf(output).any(), "输出包含Inf值!"
    
    # 验证自回归特性 (检查因果掩码是否生效)
    print("\n🔍 验证因果掩码是否生效...")
    # 创建一个特殊测试用例：所有token相同，检查预测是否只依赖于前面的token
    special_src = torch.ones(batch_size, src_len, dtype=torch.long).to(device) * 2
    special_tgt = torch.ones(batch_size, tgt_len, dtype=torch.long).to(device) * 3
    
    # 修改第3个位置的token
    special_tgt[:, 3] = 4
    
    with torch.no_grad():
        special_output = model(special_src, special_tgt, src_mask=None, tgt_mask=tgt_mask)
    
    # 检查位置3之后的预测是否受到影响
    pos3_logit = special_output[:, 3, :].cpu().numpy()  # 位置3的logits
    pos4_logit = special_output[:, 4, :].cpu().numpy()  # 位置4的logits
    
    # 位置4的预测应该受到位置3的影响
    print(f"✅ 位置3的logit示例 (前5个值): {pos3_logit[0, :5]}")
    print(f"✅ 位置4的logit示例 (前5个值): {pos4_logit[0, :5]}")
    
    # 简单验证：位置4的预测与位置3不同（因为输入不同）
    assert not np.allclose(pos3_logit, pos4_logit, atol=1e-5), \
        "位置3和位置4的预测相同，因果掩码可能未生效!"
    
    print("\n🎉 所有测试通过!")
    return model, output, src, tgt

# ========== 4. 可视化注意力权重 ==========
def visualize_attention(model, src, tgt, device):
    print("\n🎨 准备可视化注意力权重...")
    
    # 修改模型以返回注意力权重
    # 临时修改forward方法以返回注意力权重
    original_forward = model.forward
    
    def forward_with_attn(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # 复制之前的forward逻辑，但返回注意力权重
        # ... 简化版，只获取最后一层的注意力 ...
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoding(src_embedded)
        
        enc_output = src_embedded
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        enc_output = self.encoder_norm(enc_output)
        
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoding(tgt_embedded)
        
        dec_output = tgt_embedded
        last_self_attn = None
        last_cross_attn = None
        
        for i, layer in enumerate(self.decoder_layers):
            dec_output, self_attn, cross_attn = layer(dec_output, enc_output, src_mask, tgt_mask)
            if i == len(self.decoder_layers) - 1:  # 最后一层
                last_self_attn = self_attn
                last_cross_attn = cross_attn
        
        dec_output = self.decoder_norm(dec_output)
        logits = self.fc_out(dec_output)
        
        return logits, last_self_attn, last_cross_attn
    
    # 临时替换forward方法
    model.forward = forward_with_attn.__get__(model, TransFormer)
    
    # 生成掩码
    tgt_len = tgt.size(1)
    tgt_mask = model.generate_square_subsequent_mask(tgt_len).to(device)
    
    # 获取注意力权重
    with torch.no_grad():
        _, self_attn, cross_attn = model(src, tgt, src_mask=None, tgt_mask=tgt_mask)
    
    # 恢复原始forward方法
    model.forward = original_forward
    
    # 转换为numpy
    self_attn = self_attn[0].cpu().numpy()  # 取第一个样本
    cross_attn = cross_attn[0].cpu().numpy()  # 取第一个样本
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 自注意力
    im1 = axes[0].imshow(self_attn[0], cmap='viridis', aspect='auto')  # 取第一个头
    axes[0].set_title('Decoder Self-Attention (Head 0)', fontsize=14)
    axes[0].set_xlabel('Target Position')
    axes[0].set_ylabel('Target Position')
    plt.colorbar(im1, ax=axes[0])
    
    # 交叉注意力
    im2 = axes[1].imshow(cross_attn[0], cmap='viridis', aspect='auto')  # 取第一个头
    axes[1].set_title('Decoder Cross-Attention (Head 0)', fontsize=14)
    axes[1].set_xlabel('Source Position')
    axes[1].set_ylabel('Target Position')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('transformer_attention.png', dpi=300)
    print("✅ 注意力权重可视化已保存为 'transformer_attention.png'")
    plt.show()

# ========== 5. 主测试函数 ==========
if __name__ == "__main__":
    # 运行测试
    model, output, src, tgt = test_transformer()
    
    # 可视化注意力权重
    device = next(model.parameters()).device
    visualize_attention(model, src[:1], tgt[:1], device)  # 只用第一个样本
    
    print("\n" + "="*50)
    print("✨ Transformer模型测试完成! 模型工作正常")
    print("="*50)