import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self,dropout=0.1):
       super(ScaledDotProductAttention,self).__init__()
       self.dropout = nn.Dropout(dropout)
    
    def forward(self,query,key,value,mask=None):
        d_k = query.size(-1)
        
        # query: [batch_size, heads, seq_len_q, d_k]
        # key： [batch_size, heads, seq_len_k, d_k] 
        # key.transpose(-2,-1) -> [batch_size, heads, d_k, seq_len_k]
        # torch.matmul(query,key.transpose(-2,-1)) -> [batch_size, heads, seq_len_q, seq_len_k]
        # scores: [batch_size, heads, seq_len_q, seq_len_k]
        # 计算了 query 中每个位置与 key 中每个位置之间的相似度或"关联度
        # 1. 查询序列中的每个位置（token）都被表示为一个 d_k 维向量
        # 2. 键序列中的每个位置也被表示为一个 d_k 维向量
        # 3. 通过点积计算查询序列中每个位置与键序列中每个位置之间的相似度
        # 当特征维度 d_k 较大时，点积的方差也会变大，导致 softmax 函数梯度变得非常小（饱和）。
        # 通过除以 √d_k，我们将方差控制在合理范围内，使梯度更稳定
        scores = torch.matmul(query,key.transpose(-2,-1)) / math.sqrt(d_k)
        
        # 掩码（mask）用于控制哪些位置可以相互关注。当 mask==0 时，对应位置的注意力分数被设置为负无穷大 -inf
        
        if mask is not None:
            # mask==0 means the position is not allowed to attend to, so we set the score to -inf
            scores = scores.masked_fill(mask==0,float('-inf'))
        
        # Softmax 将注意力分数转换为概率分布，使所有分数总和为 1
        # 对于被掩码的位置（值为 -inf），softmax 后的值接近 0，意味着这些位置不会被关注
        # softmax 沿着最后一个维度（dim=-1，即 seq_len_k）计算，确保对每个查询位置，其对所有键位置的注意力权重总和为1
        attn = F.softmax(scores,dim=-1)
        # Dropout 随机将一部分注意力权重设为0，然后对剩余权重进行缩放，使总和仍为1
        # 防止过拟合
        # 在训练阶段，dropout 会随机丢弃一些注意力权重，以增强模型的泛化能力
        attn = self.dropout(attn)
        # 这一步将计算好的注意力权重 attn 与值（value）进行矩阵乘法，得到加权的值向量
        # attn 形状：[batch_size, heads, seq_len_q, seq_len_k]
        # value 形状：[batch_size, heads, seq_len_k, d_k]
        # output 形状：[batch_size, heads, seq_len_q, d_k]
        # 这相当于每个查询位置根据其对不同键位置的注意力权重，对相应的值向量进行加权平均。
        # 权重越高的位置，其对应的值向量对输出的贡献越大
        output = torch.matmul(attn,value)
        return output, attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads,d_model,dropout = 0.1):
        super().__init__()
        # heads 是多头注意力机制中的头数，d_model 是模型的维度
        # 先确保模型维度 d_model 能被头数 heads 整除，这是必要的数学约束
        assert d_model % heads == 0
        #d_k 表示每个注意力头处理的特征维度
        self.d_k = d_model // heads
        self.heads = heads
        # 线性变换将输入的 d_model 维度映射到 d_model 维度
        self.linear_query = nn.Linear(d_model,d_model)
        self.linear_key = nn.Linear(d_model,d_model)
        self.linear_value = nn.Linear(d_model,d_model)
        self.linear_out = nn.Linear(d_model,d_model)
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,query, key, value, mask = None):
        batch_size = query.size(0)
        # 1. 线性投影：self.linear_query(query) 将输入的 query 映射到 d_model 维度
        ### query: [batch_size, seq_len_q, d_model] --> [batch_size, seq_len_q, d_model]
        ### 这一步没有改变形状，但改变了向量的内容，为之后的分割做准备
        # 2. 变形：view(batch_size, -1, self.heads, self.d_k) 将 query 的形状变为 [batch_size, seq_len_q, heads, d_k]
        ### 这一步将最后一个维度 d_model 分割成两个维度：heads 和 d_k
        ### 这样每个注意力头就可以独立地处理 d_k 维度的向量
        ### -1 参数让 PyTorch 自动计算 seq_len 维度的大小
        # 3. 转置：transpose(1, 2) 将形状变为 [batch_size, heads, seq_len_q, d_k]
        ### 交换了第1维和第2维，使得同一个"头"的所有数据放在一起处理
        
        ##尽管只看到了一次线性变换的调用，但在这个调用内部，其权重矩阵（d_model,d_model）的不同部分（不同的列区间）实际上承担了各自不同头的投影任务。
        # 这些不同部分的参数是独立学习的，因此每个头会捕获不同的特征信息。
        query = self.linear_query(query).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        key  = self.linear_key(key).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        value = self.linear_value(value).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        
        # 原始掩码的形状通常是 [batch_size, seq_len_q, seq_len_k]。
        # 添加一个维度后变成 [batch_size, 1, seq_len_q, seq_len_k]，
        # 可以通过广播机制应用到多头注意力的每个头上
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        # 4. 计算注意力：调用 ScaledDotProductAttention 的 forward 方法，计算注意力
        ### query, key, value 的形状都是 [batch_size, heads, seq_len_q, d_k]
        ### mask 的形状是 [batch_size, 1, seq_len_q, seq_len_k]，用于控制注意力的计算
        ### x 的形状是 [batch_size, heads, seq_len_q, d_k]，表示每个头的输出
        x, attn = self.attention(query, key, value, mask=mask)
        # 5. 合并多头输出：
        # 将 x 的形状从 [batch_size, heads, seq_len_q, d_k] 变为 [batch_size, seq_len_q, heads * d_k]
        ###  维度交换：.transpose(1, 2) 将第1维和第2维交换，变为 [batch_size, seq_len_q, heads, d_k]
        ###  内存重排：.contiguous() 确保内存连续性，以便进行 reshape 操作
        ### 维度合并：.view(batch_size, -1, self.heads * self.d_k) 将最后两个维度合并，变为 [batch_size, seq_len_q, heads * d_k]
        ### 这样每个注意力头的输出就被拼接在一起，形成一个新的表示
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.d_k)
        
        # 6. 最终线性变换：self.linear_out(x) 将合并后的输出映射回原始的 d_model 维度
        ### 这一步将多头注意力的输出转换为与输入相同的维度，便于后续处理
        return self.linear_out(x)

class FeedForward(nn.Module):
    def __ini__(self, d_model,d_ff, dropout = 0.1):
        super(FeedForward,self).__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)
        # 实际上论文并没有确切地提到在这个模块使用 dropout
        #self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        return self.linear2(F.relu(self.linear1(x)))    #self.linear2(self.dropout(F.relu(self.linear1(x))))




#位置编码是为了让模型能够理解序列中单词的顺序和相对位置       
#利用正弦和余弦函数生成周期性变化的向量，使得每个位置都有一个唯一的编码，且不同位置之间的相对信息能够从编码中推断出来
##构造一个固定大小的编码矩阵 pe。
##使用不同频率的正弦和余弦填充不同维度。
##将生成的编码加到输入嵌入上，之后再通过 Dropout 防止过拟合。
class PositionalEncoding(nn.Module):
    #d_model：表示模型中嵌入的维度。后面位置编码的每一行的长度就是这个值。
    #dropout：Dropout 概率，用于防止过拟合。
    #max_len：表示支持的最大序列长度，默认设置为 5000，这样就可以预先生成足够长的位置信息。
    def  __int__(self, d_model, dropout = 0.1, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 生成位置编码矩阵 pe
        ## pe 的形状是 [max_len, d_model]，表示每个位置的编码向量
        pe = torch.zeros(max_len, d_model)
        
        # 计算位置编码
        ## position 的形状是 [max_len, 1]，表示每个位置的索引,便于后续与 div_term 做广播运算
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        #  计算角度因子
        ## torch.arange(0, d_model, 2) 生成一个数组，取 0、2、4、…，即偶数位置索引
        ## 每个偶数位置索引用于计算正弦函数的周期因子。
        ## 公式中的 -math.log(10000.0) / d_model 是一个缩放因子，用来控制周期的变化速度。
        ## torch.exp(...) 则计算指数，得到每个维度对应的缩放因子（称为 div_term），
        ## 这使得不同维度的周期呈指数级变化，从而使得编码在不同维度上具有不同的频率
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        # 对于矩阵 pe 的偶数列（索引 0,2,4,...），使用正弦函数填充
        # 对于奇数列（索引 1,3,5,...），使用余弦函数填充
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加一个维度，使得 pe 的形状变为 [1, max_len, d_model]
        # 这样在后续与输入数据相加时，可以直接广播到每个 batch 上
        pe = pe.unsqueeze(0)
        # 将 pe 注册为模型的一个缓冲区。这意味着 pe 会随模型保存和加载，但它不是模型的可训练参数，不会在训练过程中被更新
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x 的形状是 [batch_size, seq_len, d_model]
        # pe 的形状是 [1, max_len, d_model]，通过广播机制加到 x 上
        # 这样每个位置都有一个对应的正弦余弦编码
        # 从位置编码张量 self.pe 中，选取所有 batch（第一个维度）的数据，并在位置维度（第二个维度）
        # 上取前 seq_len 个编码，而第三个维度保持不变
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
        
        
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        
        ## 1. 先进行 LayerNorm，确保输入的每个位置的特征均值为0，方差为1
        ## 2. 然后进行多头自注意力计算，得到新的表示
        ## 3. 将自注意力的输出与原始输入相加，形成残差连接
        ## 4. 再进行 LayerNorm，确保每个位置的特征均值为0，方差为1
        ## 5. 经过前馈网络计算，得到新的表示
        ## 6. 将前馈网络的输出与原始输入相加，形成残差连接
        
        x2 = self.norm1(x)
        
        x = x + self.dropout(self.self_attn(x2, x2, x2, mask=mask))
        
        x2 = self.norm2(x)
        x = x + self.dropout(self.feed_forward(x2))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.cross_attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout(self.self_attn(x2, x2, x2, mask=tgt_mask))
        
        x2 = self.norm2(x)
        x = x + self.dropout(self.cross_attn(x2, memory, memory, mask=src_mask))
        
        x2 = self.norm3(x)
        x = x + self.dropout(self.feed_forward(x2))
        return x
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, dropout=0.1):
        super(Encoder,self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, src, mask = None):
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)
    

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, dropout=0.1):
        super(Decoder,self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self,tgt,memory,src_mask=None,tgt_mask=None):
        x = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x,memory,src_mask=src_mask,tgt_mask=tgt_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, dropout=0.1):
        super(Transformer,self).__init__()
        self.encoder = Encoder(vocab_size, d_model, N, heads, d_ff, dropout)
        self.decoder = Decoder(vocab_size, d_model, N, heads, d_ff, dropout)
        self.linear = nn.Linear(d_model,vocab_size)
    def forward(self, src,tgt,src_mask=None,tgt_mask=None):
        memory = self.encoder(src,mask=src_mask)
        output = self.decoder(tgt,memory,src_mask=src_mask,tgt_mask=tgt_mask)
        return self.linear(output)  # [batch_size, seq_len_tgt, vocab_size]
    

def subsequent_mask(size):
    # 生成一个上三角矩阵，大小为 (size, size)，上三角部分为 1，其他部分为 0
    # 这个掩码用于确保解码器在预测当前单词时，只能看到之前的单词
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).bool()
    return subsequent_mask  # [1, size, size]

if __name__ == '__main__':
    # 测试 Transformer 模型的基本结构和前向传播
    vocab_size = 10000  # 词汇表大小
    d_model = 512  # 嵌入维度
    N = 6  # 编码器和解码器的层数
    heads = 8  # 注意力头数
    d_ff = 2048  # 前馈网络的隐藏层维度
    dropout = 0.1  # Dropout 概率

    transformer = Transformer(vocab_size, d_model, N, heads, d_ff, dropout)

    src = torch.randint(0, vocab_size, (32, 10))  # [batch_size, seq_len_src]
    tgt = torch.randint(0, vocab_size, (32, 12))  # [batch_size, seq_len_tgt]

    src_mask = torch.ones(32, 1, 10).bool()  # [batch_size, 1, seq_len_src]
    tgt_mask = subsequent_mask(12).to(tgt.device)  # [1, seq_len_tgt, seq_len_tgt]

    output = transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
    print(output.shape)  # 应该是 [batch_size, seq_len_tgt, vocab_size]
        