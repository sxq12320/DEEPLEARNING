import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import random
import os

# 确保matplotlib在无GUI环境中也能工作
import matplotlib
matplotlib.use('Agg')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ========== 1. 简化数据准备：纯Python实现 ==========
class TranslationDataset(Dataset):
    """简化版翻译数据集，无需外部依赖"""
    def __init__(self, en_sentences, de_sentences, en_vocab, de_vocab, max_len=20):
        self.en_sentences = en_sentences
        self.de_sentences = de_sentences
        self.en_vocab = en_vocab
        self.de_vocab = de_vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.en_sentences)
    
    def __getitem__(self, idx):
        # 英文：添加<sos>和<eos>标记
        en_tokens = ['<sos>'] + self._tokenize(self.en_sentences[idx]) + ['<eos>']
        en_tokens = en_tokens[:self.max_len]  # 截断
        en_ids = [self.en_vocab.get(token, self.en_vocab['<unk>']) for token in en_tokens]
        
        # 德文：输入添加<sos>，目标添加<eos>
        de_tokens = self._tokenize(self.de_sentences[idx])
        de_input = ['<sos>'] + de_tokens
        de_target = de_tokens + ['<eos>']
        
        de_input = de_input[:self.max_len]
        de_target = de_target[:self.max_len]
        
        de_input_ids = [self.de_vocab.get(token, self.de_vocab['<unk>']) for token in de_input]
        de_target_ids = [self.de_vocab.get(token, self.de_vocab['<unk>']) for token in de_target]
        
        # 创建掩码
        en_mask = torch.ones(len(en_ids), dtype=torch.bool)
        tgt_len = len(de_input_ids)
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len))
        
        # 转换为张量
        return {
            'src': torch.tensor(en_ids),
            'tgt': torch.tensor(de_input_ids),
            'tgt_y': torch.tensor(de_target_ids),
            'src_mask': en_mask,
            'tgt_mask': tgt_mask,
            'ntokens': len(de_target_ids)
        }
    
    def _tokenize(self, text):
        """简单的空格分词，不依赖spacy"""
        # 移除标点，转换为小写
        text = text.lower()
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        return text.split()

def collate_fn(batch):
    """处理变长序列的批处理"""
    max_src_len = max(len(item['src']) for item in batch)
    max_tgt_len = max(len(item['tgt']) for item in batch)
    
    src_batch = []
    tgt_batch = []
    tgt_y_batch = []
    src_mask_batch = []
    tgt_mask_batch = []
    ntokens = 0
    
    for item in batch:
        # 源序列填充
        src_padded = torch.cat([
            item['src'],
            torch.zeros(max_src_len - len(item['src']), dtype=torch.long)
        ])
        src_batch.append(src_padded)
        
        # 目标序列填充
        tgt_padded = torch.cat([
            item['tgt'],
            torch.zeros(max_tgt_len - len(item['tgt']), dtype=torch.long)
        ])
        tgt_batch.append(tgt_padded)
        
        # 目标y填充
        tgt_y_padded = torch.cat([
            item['tgt_y'],
            torch.zeros(max_tgt_len - len(item['tgt_y']), dtype=torch.long)
        ])
        tgt_y_batch.append(tgt_y_padded)
        
        # 源掩码 (1表示真实token，0表示padding)
        src_mask = torch.cat([
            item['src_mask'],
            torch.zeros(max_src_len - len(item['src_mask']), dtype=torch.bool)
        ])
        src_mask_batch.append(src_mask)
        
        # 目标掩码 (下三角矩阵)
        tgt_mask_square = torch.zeros(max_tgt_len, max_tgt_len)
        tgt_mask_square[:len(item['tgt_mask']), :len(item['tgt_mask'])] = item['tgt_mask']
        tgt_mask_batch.append(tgt_mask_square)
        
        ntokens += item['ntokens']
    
    return {
        'src': torch.stack(src_batch),
        'tgt': torch.stack(tgt_batch),
        'tgt_y': torch.stack(tgt_y_batch),
        'src_mask': torch.stack(src_mask_batch),
        'tgt_mask': torch.stack(tgt_mask_batch),
        'ntokens': ntokens
    }

# 创建小型英德数据集
def create_small_dataset():
    """创建小型英德翻译数据集，无需外部文件"""
    data = [
        ("I love machine learning", "Ich liebe maschinelles Lernen"),
        ("This is a test sentence", "Dies ist ein Testsatz"),
        ("Hello world", "Hallo Welt"),
        ("How are you today", "Wie geht es dir heute"),
        ("The weather is nice", "Das Wetter ist schön"),
        ("I want to learn German", "Ich möchte Deutsch lernen"),
        ("She is reading a book", "Sie liest ein Buch"),
        ("We go to school every day", "Wir gehen jeden Tag zur Schule"),
        ("The cat is on the table", "Die Katze ist auf dem Tisch"),
        ("This restaurant is very good", "Dieses Restaurant ist sehr gut"),
        ("I have two brothers", "Ich habe zwei Brüder"),
        ("Can you help me please", "Kannst du mir bitte helfen"),
        ("What time is it", "Wie spät ist es"),
        ("I like to play football", "Ich spiele gerne Fußball"),
        ("The train is late", "Der Zug hat Verspätung"),
        ("I am hungry", "Ich habe Hunger"),
        ("Where is the bathroom", "Wo ist die Toilette"),
        ("It is raining outside", "Es regnet draußen"),
        ("I speak English and German", "Ich spreche Englisch und Deutsch"),
        ("This is my favorite song", "Das ist mein Lieblingslied"),
        ("Artificial intelligence is fascinating", "Künstliche Intelligenz ist faszinierend"),
        ("Let's go for a walk", "Gehen wir spazieren"),
        ("I need a coffee", "Ich brauche einen Kaffee"),
        ("The meeting is at 3 PM", "Das Meeting ist um 15 Uhr"),
        ("Have a nice day", "Einen schönen Tag noch"),
        ("Thank you very much", "Vielen Dank"),
        ("You are welcome", "Gern geschehen"),
        ("Excuse me", "Entschuldigung"),
        ("I don't understand", "Ich verstehe nicht"),
        ("Could you repeat that", "Könnten Sie das wiederholen"),
        ("Where can I find a hotel", "Wo kann ich ein Hotel finden"),
        ("How much does it cost", "Wie viel kostet es"),
        ("I would like to order", "Ich möchte bestellen"),
        ("The food is delicious", "Das Essen ist köstlich"),
        ("I'm lost", "Ich habe mich verlaufen"),
        ("Call the police", "Rufen Sie die Polizei"),
        ("I need a doctor", "Ich brauche einen Arzt"),
        ("It's an emergency", "Es ist ein Notfall"),
        ("Happy birthday", "Alles Gute zum Geburtstag"),
        ("Congratulations", "Herzlichen Glückwunsch")
    ]
    
    # 随机打乱数据
    random.shuffle(data)
    
    en_sentences = [pair[0].lower() for pair in data]
    de_sentences = [pair[1].lower() for pair in data]
    
    return en_sentences, de_sentences

# 构建词汇表
def build_vocab(sentences, max_vocab_size=500):
    """从句子列表构建词汇表，纯Python实现"""
    word_count = Counter()
    for sentence in sentences:
        # 简单分词
        words = sentence.lower()
        words = ''.join(c for c in words if c.isalnum() or c.isspace())
        words = words.split()
        word_count.update(words)
    
    # 保留最常见的单词
    vocab_words = ['<pad>', '<sos>', '<eos>', '<unk>'] + [word for word, count in word_count.most_common(max_vocab_size-4)]
    
    vocab = {word: idx for idx, word in enumerate(vocab_words)}
    return vocab

# ========== 2. 模型定义 (修复所有关键错误) ==========
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        out_matmul = torch.matmul(Q, K.transpose(-2, -1))
        dk = K.size(-1)
        out_scaled = out_matmul / math.sqrt(dk)

        if mask is not None:
            if mask.dim() == 2:  # [seq_q, seq_k]
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_q, seq_k]
            elif mask.dim() == 3:  # [batch, seq_q, seq_k] 
                mask = mask.unsqueeze(1)  # [batch, 1, seq_q, seq_k]
            out_scaled = out_scaled.masked_fill(mask == 0, -1e9)
        
        attn = self.softmax(out_scaled)
        attn_weight = self.dropout(attn)
        output = torch.matmul(attn_weight, V)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.scale_dot_product_attention = ScaledDotProductAttention(dropout)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, len_seq, d_model = x.size()
        x = x.view(batch_size, len_seq, self.n_heads, self.d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, n_heads, len_seq, d_k = x.size()
        x = x.transpose(1, 2).contiguous().view(batch_size, len_seq, self.d_model)
        return x
    
    def forward(self, Q, K, V, mask=None):
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        output, attn = self.scale_dot_product_attention(Q, K, V, mask)
        output = self.combine_heads(output)
        return output, attn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out_fc1 = self.fc1(x)
        out_relu = self.relu(out_fc1)
        out_dropout = self.dropout(out_relu)
        output = self.fc2(out_dropout)
        return output

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = self.gamma * (x - mean) / (std + self.eps) + self.beta
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        output = self.norm2(x)
        return output

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn1 = MultiHeadAttention(d_model, n_heads, dropout)
        self.LayerNorm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.self_attn2 = MultiHeadAttention(d_model, n_heads, dropout)
        self.LayerNorm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.LayerNorm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 自注意力 - 使用目标掩码（防止看到未来词）
        attn_output1, attn_weights1 = self.self_attn1(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output1)
        x = self.LayerNorm1(x)

        # 编码器-解码器注意力
        attn_output2, attn_weights2 = self.self_attn2(x, enc_output, enc_output, src_mask)
        x = x + self.dropout2(attn_output2)
        x = self.LayerNorm2(x)

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        output = self.LayerNorm3(x)

        return output, attn_weights1, attn_weights2

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, n_heads=4, 
                 d_ff=512, num_encoder_layers=3, num_decoder_layers=3, dropout=0.1):
        super(Transformer, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = LayerNorm(d_model)
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = LayerNorm(d_model)
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.d_model = d_model
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.src_embedding.weight.data *= math.sqrt(self.d_model)
        self.tgt_embedding.weight.data *= math.sqrt(self.d_model)

    def generate_square_subsequent_mask(self, sz):
        """生成下三角掩码，防止解码器看到未来词"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 源序列处理
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoding(src_embedded)
        
        enc_output = src_embedded
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        enc_output = self.encoder_norm(enc_output)
        
        # 目标序列处理
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoding(tgt_embedded)
        
        dec_output = tgt_embedded
        for layer in self.decoder_layers:
            dec_output, _, _ = layer(dec_output, enc_output, src_mask, tgt_mask)
        dec_output = self.decoder_norm(dec_output)
        
        # 输出层
        output = self.fc_out(dec_output)
        return output

# ========== 3. 训练和推理函数 ==========
def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    total_tokens = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{15}")
    for batch in progress_bar:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        tgt_y = batch['tgt_y'].to(device)
        src_mask = batch['src_mask'].to(device)
        tgt_mask = batch['tgt_mask'].to(device)
        ntokens = batch['ntokens']
        
        optimizer.zero_grad()
        
        # 前向传播
        # 调整掩码维度: [batch_size, 1, seq_len] -> [batch_size, 1, 1, seq_len]
        src_mask_expanded = src_mask.unsqueeze(1).unsqueeze(1)
        
        # 调整目标掩码维度: [batch_size, seq_len, seq_len] -> [batch_size, 1, seq_len, seq_len]
        tgt_mask_expanded = tgt_mask.unsqueeze(1)
        
        output = model(src, tgt, src_mask_expanded, tgt_mask_expanded)
        
        # 计算损失 (忽略padding)
        loss = criterion(output.view(-1, output.size(-1)), tgt_y.view(-1))
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * ntokens
        total_tokens += ntokens
        
        # 更新进度条
        avg_loss = total_loss / total_tokens
        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
    
    return total_loss / total_tokens

def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    """贪婪解码"""
    model.eval()
    with torch.no_grad():
        # 编码源序列
        src_embedded = model.src_embedding(src) * math.sqrt(model.d_model)
        src_embedded = model.pos_encoding(src_embedded)
        
        enc_output = src_embedded
        for layer in model.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        enc_output = model.encoder_norm(enc_output)
        
        # 初始化目标序列
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src).long().to(device)
        
        # 逐步生成
        for i in range(max_len-1):
            # 生成自回归掩码
            tgt_mask = model.generate_square_subsequent_mask(ys.size(1)).type_as(src_mask)
            
            # 解码
            tgt_embedded = model.tgt_embedding(ys) * math.sqrt(model.d_model)
            tgt_embedded = model.pos_encoding(tgt_embedded)
            
            dec_output = tgt_embedded
            for layer in model.decoder_layers:
                dec_output, _, _ = layer(dec_output, enc_output, src_mask, tgt_mask.unsqueeze(0).unsqueeze(0))
            dec_output = model.decoder_norm(dec_output)
            
            prob = model.fc_out(dec_output[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            
            # 添加新token
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src).fill_(next_word).long().to(device)], dim=1)
            
            # 检查结束
            if next_word == end_symbol or ys.size(1) >= max_len:
                break
                
        return ys

def translate_sentence(model, sentence, en_vocab, de_vocab, device, max_len=20):
    """翻译单个句子，纯Python实现"""
    # 预处理
    sentence = sentence.lower()
    sentence = ''.join(c for c in sentence if c.isalnum() or c.isspace())
    
    tokens = ['<sos>'] + sentence.split() + ['<eos>']
    src_ids = [en_vocab.get(token, en_vocab['<unk>']) for token in tokens]
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)
    
    # 创建掩码
    src_mask = (src_tensor != en_vocab['<pad>']).unsqueeze(1).unsqueeze(1)
    
    # 解码
    decoded_ids = greedy_decode(
        model, 
        src_tensor, 
        src_mask, 
        max_len, 
        de_vocab['<sos>'], 
        de_vocab['<eos>'], 
        device
    )
    
    # 转换为文本
    decoded_ids = decoded_ids.squeeze().cpu().numpy()
    if decoded_ids.ndim == 0:
        decoded_ids = np.array([decoded_ids])
    
    translated_tokens = []
    for idx in decoded_ids:
        # 找到词汇表中对应的词
        token = None
        for word, word_id in de_vocab.items():
            if word_id == idx:
                token = word
                break
        
        if token is None:
            continue
            
        if token == '<eos>':
            break
        if token not in ['<sos>', '<pad>', '<unk>']:
            translated_tokens.append(token)
    
    return ' '.join(translated_tokens)

# ========== 4. 主训练流程 ==========
def main():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")
    
    # 1. 准备数据
    print("\n📚 准备小型英德翻译数据集...")
    en_sentences, de_sentences = create_small_dataset()
    print(f"✅ 数据集大小: {len(en_sentences)} 句子")
    print(f"🔍 部分数据示例:")
    for i in range(3):
        print(f"  - {en_sentences[i]} → {de_sentences[i]}")
    
    # 构建词汇表
    print("\n🔤 构建词汇表...")
    en_vocab = build_vocab(en_sentences, max_vocab_size=500)
    de_vocab = build_vocab(de_sentences, max_vocab_size=500)
    print(f"✅ 英文词汇表大小: {len(en_vocab)}")
    print(f"✅ 德文词汇表大小: {len(de_vocab)}")
    print(f"🔍 词汇表示例: {list(en_vocab.items())[:10]}")
    
    # 创建数据集和数据加载器
    dataset = TranslationDataset(en_sentences, de_sentences, en_vocab, de_vocab, max_len=15)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    # 2. 初始化模型
    print("\n🔧 初始化Transformer模型...")
    model = Transformer(
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(de_vocab),
        d_model=128,
        n_heads=4,
        d_ff=512,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dropout=0.1
    ).to(device)
    
    # 3. 训练配置
    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # 4. 训练模型
    print("\n🔥 开始训练...")
    num_epochs = 100
    losses = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device, epoch)
        end_time = time.time()
        
        losses.append(train_loss)
        print(f"✅ Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f} - Time: {end_time-start_time:.2f}s")
        
        # 每3个epoch展示一个翻译示例
        if (epoch + 1) % 3 == 0:
            test_sentence = "I love machine learning"
            translation = translate_sentence(model, test_sentence, en_vocab, de_vocab, device)
            print(f"💬 测试: '{test_sentence}' → '{translation}'")
    
    # 5. 评估模型
    print("\n🎯 评估训练好的模型...")
    test_sentences = [
        "I love machine learning",
        "The weather is nice",
        "Hello world",
        "How are you today",
        "This is a test sentence",
        "I need a coffee",
        "Where is the bathroom"
    ]
    
    print("\n📈 训练损失曲线")
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 保存图表
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_loss.png')
    print("💾 损失曲线已保存到 'results/training_loss.png'")
    plt.close()
    
    print("\n📝 翻译示例:")
    results = []
    for sentence in test_sentences:
        translation = translate_sentence(model, sentence, en_vocab, de_vocab, device)
        print(f"  🇬🇧 '{sentence}'")
        print(f"  🇩🇪 '{translation}'")
        results.append(f"'{sentence}' → '{translation}'")
        print("-" * 50)
    
    # 保存结果
    with open('results/translation_results.txt', 'w', encoding='utf-8') as f:
        f.write("Transformer 翻译结果\n")
        f.write("=" * 50 + "\n")
        for result in results:
            f.write(result + "\n")
    print("💾 翻译结果已保存到 'results/translation_results.txt'")
    
    # 6. 保存模型
    torch.save(model.state_dict(), 'results/transformer_translation_model.pth')
    print("\n💾 模型已保存为 'results/transformer_translation_model.pth'")
    
    # 7. 交互式测试
    print("\n✨ 进入交互模式！输入'exit'退出")
    while True:
        user_input = input("\n🇬🇧 请输入要翻译的英文: ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
            
        try:
            translation = translate_sentence(model, user_input, en_vocab, de_vocab, device)
            print(f"🇩🇪 翻译结果: '{translation}'")
        except Exception as e:
            print(f"❌ 翻译出错: {e}")
            print("💡 提示: 请使用简单的英文句子，避免特殊字符")

if __name__ == "__main__":
    main()