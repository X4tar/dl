"""
Transformer 组件详细实现
包含所有基础组件的详细解释和实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力机制
    
    这是 Transformer 的核心组件。它计算查询(Q)、键(K)、值(V)之间的注意力权重。
    
    公式: Attention(Q,K,V) = softmax(QK^T / √d_k)V
    """
    
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        前向传播
        
        Args:
            Q: 查询矩阵 [batch_size, seq_len, d_k]
            K: 键矩阵 [batch_size, seq_len, d_k]  
            V: 值矩阵 [batch_size, seq_len, d_v]
            mask: 掩码矩阵，用于遮蔽某些位置
            
        Returns:
            output: 注意力输出 [batch_size, seq_len, d_v]
            attention_weights: 注意力权重 [batch_size, seq_len, seq_len]
        """
        
        # 步骤1: 计算注意力分数 QK^T
        # Q: [batch_size, seq_len, d_k]
        # K: [batch_size, seq_len, d_k]
        # scores: [batch_size, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # 步骤2: 缩放（防止梯度消失）
        scores = scores / math.sqrt(self.d_k)
        
        # 步骤3: 应用掩码（如果提供）
        if mask is not None:
            # 将掩码位置设为很小的负数，这样 softmax 后会接近 0
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 步骤4: 应用 softmax 获得注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 步骤5: 加权求和
        # attention_weights: [batch_size, seq_len, seq_len]
        # V: [batch_size, seq_len, d_v]
        # output: [batch_size, seq_len, d_v]
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    多头注意力允许模型同时关注来自不同表示子空间的信息。
    它将输入投影到多个不同的子空间，在每个子空间中计算注意力，
    然后将结果连接起来。
    
    公式: MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
         head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        
        self.d_model = d_model  # 模型维度
        self.n_heads = n_heads  # 注意力头数
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)  # Query 投影
        self.W_k = nn.Linear(d_model, d_model)  # Key 投影  
        self.W_v = nn.Linear(d_model, d_model)  # Value 投影
        self.W_o = nn.Linear(d_model, d_model)  # 输出投影
        
        # 注意力机制
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: 掩码
            
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 步骤1: 线性投影并重塑为多头形式
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_heads, d_k]
        # -> [batch_size, n_heads, seq_len, d_k]
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 调整掩码维度以适应多头
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        
        # 步骤2: 计算注意力
        # attn_output: [batch_size, n_heads, seq_len, d_k]
        # attention_weights: [batch_size, n_heads, seq_len, seq_len]
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        
        # 步骤3: 连接多头结果
        # [batch_size, n_heads, seq_len, d_k] -> [batch_size, seq_len, n_heads, d_k]
        # -> [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 步骤4: 输出投影
        output = self.W_o(attn_output)
        
        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络
    
    这是一个简单的两层全连接网络，在每个位置上独立应用。
    通常第一层会扩大维度，第二层再缩回原来的维度。
    
    公式: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # x -> linear1 -> ReLU -> dropout -> linear2
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class LayerNorm(nn.Module):
    """
    层归一化
    
    层归一化对每个样本的特征维度进行归一化，
    有助于稳定训练过程并加速收敛。
    """
    
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        
        self.gamma = nn.Parameter(torch.ones(d_model))  # 缩放参数
        self.beta = nn.Parameter(torch.zeros(d_model))  # 偏移参数
        self.eps = eps
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # 计算均值和方差（在最后一个维度上）
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # 归一化
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # 应用学习的缩放和偏移
        return self.gamma * normalized + self.beta


class EncoderLayer(nn.Module):
    """
    Transformer 编码器层
    
    每个编码器层包含：
    1. 多头自注意力机制
    2. 残差连接和层归一化
    3. 位置前馈网络
    4. 残差连接和层归一化
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.multi_head_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, d_model]
            mask: 注意力掩码
            
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: 注意力权重
        """
        # 子层1: 多头自注意力 + 残差连接 + 层归一化
        attn_output, attention_weights = self.multi_head_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 子层2: 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights


class DecoderLayer(nn.Module):
    """
    Transformer 解码器层
    
    每个解码器层包含：
    1. 掩码多头自注意力机制
    2. 残差连接和层归一化
    3. 编码器-解码器注意力机制
    4. 残差连接和层归一化
    5. 位置前馈网络
    6. 残差连接和层归一化
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.masked_multi_head_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.encoder_decoder_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        前向传播
        
        Args:
            x: 解码器输入 [batch_size, tgt_seq_len, d_model]
            encoder_output: 编码器输出 [batch_size, src_seq_len, d_model]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码（用于掩码自注意力）
            
        Returns:
            output: [batch_size, tgt_seq_len, d_model]
            self_attention_weights: 自注意力权重
            cross_attention_weights: 交叉注意力权重
        """
        # 子层1: 掩码多头自注意力 + 残差连接 + 层归一化
        self_attn_output, self_attention_weights = self.masked_multi_head_attention(
            x, x, x, tgt_mask
        )
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 子层2: 编码器-解码器注意力 + 残差连接 + 层归一化
        cross_attn_output, cross_attention_weights = self.encoder_decoder_attention(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 子层3: 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, self_attention_weights, cross_attention_weights


def create_padding_mask(seq, pad_token=0):
    """
    创建填充掩码
    
    Args:
        seq: 输入序列 [batch_size, seq_len]
        pad_token: 填充标记的值
        
    Returns:
        mask: [batch_size, 1, 1, seq_len]
    """
    mask = (seq != pad_token).unsqueeze(1).unsqueeze(2)
    return mask


def create_look_ahead_mask(size):
    """
    创建前瞻掩码（用于解码器的自注意力）
    
    Args:
        size: 序列长度
        
    Returns:
        mask: [size, size] 下三角矩阵
    """
    mask = torch.tril(torch.ones(size, size))
    return mask


if __name__ == "__main__":
    # 测试各个组件
    print("测试 Transformer 组件...")
    
    # 设置参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    d_ff = 2048
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"输入形状: {x.shape}")
    
    # 测试多头注意力
    print("\n1. 测试多头注意力...")
    multi_head_attn = MultiHeadAttention(d_model, n_heads)
    attn_output, attn_weights = multi_head_attn(x, x, x)
    print(f"注意力输出形状: {attn_output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    
    # 测试前馈网络
    print("\n2. 测试前馈网络...")
    ff = PositionwiseFeedForward(d_model, d_ff)
    ff_output = ff(x)
    print(f"前馈网络输出形状: {ff_output.shape}")
    
    # 测试编码器层
    print("\n3. 测试编码器层...")
    encoder_layer = EncoderLayer(d_model, n_heads, d_ff)
    enc_output, enc_attn_weights = encoder_layer(x)
    print(f"编码器输出形状: {enc_output.shape}")
    
    # 测试解码器层
    print("\n4. 测试解码器层...")
    decoder_layer = DecoderLayer(d_model, n_heads, d_ff)
    dec_output, self_attn_weights, cross_attn_weights = decoder_layer(x, enc_output)
    print(f"解码器输出形状: {dec_output.shape}")
    
    print("\n所有组件测试完成！")
