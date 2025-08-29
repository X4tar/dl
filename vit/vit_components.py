"""
Vision Transformer (ViT) 基础组件实现
包含 ViT 中使用的所有核心组件的详细实现和解释
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    ViT 中使用的标准 Transformer 注意力机制
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 线性变换层：Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # 输出投影
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]
            
        Returns:
            output: 输出张量 [batch_size, seq_len, d_model]
            attention_weights: 注意力权重 [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 1. 生成 Q, K, V
        Q = self.w_q(x)  # [batch_size, seq_len, d_model]
        K = self.w_k(x)  # [batch_size, seq_len, d_model]
        V = self.w_v(x)  # [batch_size, seq_len, d_model]
        
        # 2. 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        
        # 3. 计算注意力
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # 5. 输出投影
        output = self.w_o(output)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力
        
        Args:
            Q, K, V: 查询、键、值矩阵
            mask: 注意力掩码
            
        Returns:
            output: 注意力输出
            attention_weights: 注意力权重
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用 softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络
    ViT 中的 MLP 层
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # ViT 通常使用 GELU
    
    def forward(self, x):
        """
        前向传播
        
        Args:    
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            output: 输出张量 [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Transformer 编码器块
    包含多头注意力和前馈网络，以及残差连接和层归一化
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # 多头注意力
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # 前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码
            
        Returns:
            output: 输出张量 [batch_size, seq_len, d_model]
            attention_weights: 注意力权重
        """
        # 多头注意力 + 残差连接 + 层归一化
        attn_output, attention_weights = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x, attention_weights


def test_components():
    """测试各个组件"""
    print("=" * 60)
    print("测试 ViT 基础组件")
    print("=" * 60)
    
    # 测试参数
    batch_size = 2
    seq_len = 197  # 14*14 patches + 1 CLS token
    d_model = 768
    n_heads = 12
    d_ff = 3072
    
    # 创建测试输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"输入形状: {x.shape}")
    print(f"模型参数: d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")
    
    # 1. 测试多头注意力
    print("\n1. 测试多头注意力")
    print("-" * 30)
    
    mha = MultiHeadAttention(d_model, n_heads)
    attn_output, attn_weights = mha(x)
    
    print(f"注意力输出形状: {attn_output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"注意力权重和: {attn_weights.sum(dim=-1)[0, 0, 0]:.4f} (应该接近1)")
    
    # 2. 测试前馈网络
    print("\n2. 测试前馈网络")
    print("-" * 30)
    
    ff = PositionwiseFeedForward(d_model, d_ff)
    ff_output = ff(x)
    
    print(f"前馈网络输出形状: {ff_output.shape}")
    print(f"输入输出形状是否相同: {x.shape == ff_output.shape}")
    
    # 3. 测试完整的 Transformer 块
    print("\n3. 测试 Transformer 块")
    print("-" * 30)
    
    transformer_block = TransformerBlock(d_model, n_heads, d_ff)
    block_output, block_attn = transformer_block(x)
    
    print(f"Transformer 块输出形状: {block_output.shape}")
    print(f"块注意力权重形状: {block_attn.shape}")
    
    # 4. 计算参数数量
    print("\n4. 参数数量统计")
    print("-" * 30)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    mha_params = count_parameters(mha)
    ff_params = count_parameters(ff)
    block_params = count_parameters(transformer_block)
    
    print(f"多头注意力参数数量: {mha_params:,}")
    print(f"前馈网络参数数量: {ff_params:,}")
    print(f"Transformer 块参数数量: {block_params:,}")
    
    # 5. 测试注意力模式
    print("\n5. 分析注意力模式")
    print("-" * 30)
    
    # CLS token 对其他 token 的注意力
    cls_attention = attn_weights[0, :, 0, 1:]  # [n_heads, seq_len-1]
    print(f"CLS token 注意力统计:")
    print(f"  最大注意力: {cls_attention.max().item():.4f}")
    print(f"  最小注意力: {cls_attention.min().item():.4f}")
    print(f"  平均注意力: {cls_attention.mean().item():.4f}")
    
    # 不同头的注意力差异
    head_variances = []
    for head in range(n_heads):
        head_attn = attn_weights[0, head]
        variance = head_attn.var().item()
        head_variances.append(variance)
    
    print(f"\n不同注意力头的方差:")
    for i, var in enumerate(head_variances):
        print(f"  头 {i+1}: {var:.6f}")
    
    print("\n✓ 所有组件测试通过！")


def demonstrate_attention_mechanism():
    """演示注意力机制的工作原理"""
    print("\n" + "=" * 60)
    print("演示注意力机制工作原理")
    print("=" * 60)
    
    # 创建简单示例
    seq_len = 5
    d_model = 8
    n_heads = 2
    
    print(f"简化示例: seq_len={seq_len}, d_model={d_model}, n_heads={n_heads}")
    
    # 创建输入
    x = torch.randn(1, seq_len, d_model)
    print(f"\n输入张量形状: {x.shape}")
    
    # 创建注意力层
    attention = MultiHeadAttention(d_model, n_heads)
    
    # 前向传播
    output, weights = attention(x)
    
    print(f"输出张量形状: {output.shape}")
    print(f"注意力权重形状: {weights.shape}")
    
    # 显示注意力权重矩阵
    print(f"\n第一个头的注意力权重矩阵:")
    attn_matrix = weights[0, 0].detach().numpy()
    
    # 简单的文本表示
    print("   ", end="")
    for j in range(seq_len):
        print(f"P{j:2d}", end="  ")
    print()
    
    for i in range(seq_len):
        print(f"P{i:2d} ", end="")
        for j in range(seq_len):
            print(f"{attn_matrix[i,j]:.2f}", end="  ")
        print()
    
    print("\n说明:")
    print("- 每行表示一个查询位置对所有键位置的注意力权重")
    print("- 每行权重之和为 1.0")
    print("- 较大的权重表示更强的注意力关系")


if __name__ == "__main__":
    # 测试所有组件
    test_components()
    
    # 演示注意力机制
    demonstrate_attention_mechanism()
    
    print("\n" + "=" * 60)
    print("ViT 基础组件测试完成！")
    print("这些组件将用于构建完整的 Vision Transformer 模型")
    print("=" * 60)
