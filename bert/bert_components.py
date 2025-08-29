"""
BERT 核心组件实现
包含 BERT 的所有基础组件：嵌入层、编码器、预训练任务等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class BERTEmbeddings(nn.Module):
    """
    BERT 嵌入层
    组合了 Token Embeddings + Segment Embeddings + Position Embeddings
    """
    
    def __init__(self, vocab_size, hidden_size, max_position_embeddings=512, 
                 type_vocab_size=2, dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        
        # Token Embeddings：词汇表中每个词的嵌入
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # Position Embeddings：学习的位置嵌入（不同于 Transformer 的固定位置编码）
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Token Type Embeddings：区分句子 A 和句子 B
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        # Layer Normalization 和 Dropout
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # 注册位置 id 缓冲区
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
    
    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len] 输入词汇 ID
            token_type_ids: [batch_size, seq_len] 句子类型 ID (0 或 1)
            position_ids: [batch_size, seq_len] 位置 ID
            
        Returns:
            embeddings: [batch_size, seq_len, hidden_size] 组合嵌入
        """
        batch_size, seq_len = input_ids.shape
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # 获取各种嵌入
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # 组合所有嵌入
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        
        # 应用 LayerNorm 和 Dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BERTSelfAttention(nn.Module):
    """
    BERT 自注意力机制
    实现多头自注意力，与标准 Transformer 相同
    """
    
    def __init__(self, hidden_size, num_attention_heads, dropout=0.1):
        super().__init__()
        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) 必须能被 num_attention_heads ({num_attention_heads}) 整除")
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Q, K, V 投影层
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def transpose_for_scores(self, x):
        """
        调整张量形状以便进行多头注意力计算
        [batch_size, seq_len, hidden_size] -> [batch_size, num_heads, seq_len, head_size]
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None, return_attention=False):
        """
        前向传播
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] 输入序列
            attention_mask: [batch_size, seq_len] 注意力掩码
            return_attention: 是否返回注意力权重
            
        Returns:
            context_layer: [batch_size, seq_len, hidden_size] 上下文表示
            attention_probs: [batch_size, num_heads, seq_len, seq_len] 注意力权重（可选）
        """
        # 计算 Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 应用注意力掩码
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        
        # 计算注意力概率
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 应用注意力权重
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # 重新调整形状
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        if return_attention:
            return context_layer, attention_probs
        else:
            return context_layer


class BERTSelfOutput(nn.Module):
    """
    BERT 自注意力输出层
    包含线性投影、残差连接和层归一化
    """
    
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states, input_tensor):
        """
        前向传播
        
        Args:
            hidden_states: 自注意力的输出
            input_tensor: 自注意力的输入（用于残差连接）
            
        Returns:
            hidden_states: 经过投影、残差连接和归一化的输出
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BERTAttention(nn.Module):
    """
    完整的 BERT 注意力模块
    包含自注意力和输出投影
    """
    
    def __init__(self, hidden_size, num_attention_heads, dropout=0.1):
        super().__init__()
        self.self = BERTSelfAttention(hidden_size, num_attention_heads, dropout)
        self.output = BERTSelfOutput(hidden_size, dropout)
    
    def forward(self, hidden_states, attention_mask=None, return_attention=False):
        """前向传播"""
        if return_attention:
            self_outputs, attention_probs = self.self(hidden_states, attention_mask, return_attention=True)
            attention_output = self.output(self_outputs, hidden_states)
            return attention_output, attention_probs
        else:
            self_outputs = self.self(hidden_states, attention_mask)
            attention_output = self.output(self_outputs, hidden_states)
            return attention_output


class BERTIntermediate(nn.Module):
    """
    BERT 中间层（前馈网络的第一部分）
    """
    
    def __init__(self, hidden_size, intermediate_size=3072):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = F.gelu  # BERT 使用 GELU 激活函数
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BERTOutput(nn.Module):
    """
    BERT 输出层（前馈网络的第二部分）
    """
    
    def __init__(self, intermediate_size, hidden_size, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BERTLayer(nn.Module):
    """
    单个 BERT 编码器层
    包含注意力机制和前馈网络
    """
    
    def __init__(self, hidden_size, num_attention_heads, intermediate_size=3072, dropout=0.1):
        super().__init__()
        self.attention = BERTAttention(hidden_size, num_attention_heads, dropout)
        self.intermediate = BERTIntermediate(hidden_size, intermediate_size)
        self.output = BERTOutput(intermediate_size, hidden_size, dropout)
    
    def forward(self, hidden_states, attention_mask=None, return_attention=False):
        """
        前向传播
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] 输入
            attention_mask: 注意力掩码
            return_attention: 是否返回注意力权重
            
        Returns:
            layer_output: 编码器层输出
            attention_probs: 注意力权重（可选）
        """
        if return_attention:
            attention_output, attention_probs = self.attention(
                hidden_states, attention_mask, return_attention=True
            )
        else:
            attention_output = self.attention(hidden_states, attention_mask)
        
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        if return_attention:
            return layer_output, attention_probs
        else:
            return layer_output


class BERTEncoder(nn.Module):
    """
    BERT 编码器
    包含多层 BERT 编码器层
    """
    
    def __init__(self, hidden_size, num_hidden_layers, num_attention_heads, 
                 intermediate_size=3072, dropout=0.1):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        
        self.layers = nn.ModuleList([
            BERTLayer(hidden_size, num_attention_heads, intermediate_size, dropout)
            for _ in range(num_hidden_layers)
        ])
    
    def forward(self, hidden_states, attention_mask=None, return_attention=False):
        """
        前向传播
        
        Args:
            hidden_states: 嵌入层输出
            attention_mask: 注意力掩码
            return_attention: 是否返回所有层的注意力权重
            
        Returns:
            sequence_output: 最后一层的输出
            all_attention_probs: 所有层的注意力权重（可选）
        """
        all_attention_probs = []
        
        for layer in self.layers:
            if return_attention:
                hidden_states, attention_probs = layer(
                    hidden_states, attention_mask, return_attention=True
                )
                all_attention_probs.append(attention_probs)
            else:
                hidden_states = layer(hidden_states, attention_mask)
        
        if return_attention:
            return hidden_states, all_attention_probs
        else:
            return hidden_states


class BERTPooler(nn.Module):
    """
    BERT 池化层
    用于从序列表示中提取句子级表示
    """
    
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states):
        """
        前向传播
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] 序列表示
            
        Returns:
            pooled_output: [batch_size, hidden_size] 句子级表示
        """
        # 取 [CLS] token 的表示（第一个位置）
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def test_bert_components():
    """测试 BERT 各个组件"""
    print("=" * 60)
    print("测试 BERT 核心组件")
    print("=" * 60)
    
    # 设置参数
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    hidden_size = 768
    num_attention_heads = 12
    num_hidden_layers = 2
    
    # 创建假数据
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"输入数据形状:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  token_type_ids: {token_type_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    
    # 1. 测试嵌入层
    print("\n1. 测试 BERT 嵌入层")
    print("-" * 30)
    
    embeddings = BERTEmbeddings(vocab_size, hidden_size)
    embedded_output = embeddings(input_ids, token_type_ids)
    
    print(f"嵌入层输出形状: {embedded_output.shape}")
    print(f"嵌入层统计: 均值={embedded_output.mean():.4f}, 标准差={embedded_output.std():.4f}")
    
    # 2. 测试自注意力
    print("\n2. 测试 BERT 自注意力")
    print("-" * 30)
    
    attention = BERTSelfAttention(hidden_size, num_attention_heads)
    attention_output, attention_probs = attention(embedded_output, attention_mask, return_attention=True)
    
    print(f"注意力输出形状: {attention_output.shape}")
    print(f"注意力权重形状: {attention_probs.shape}")
    print(f"注意力权重和: {attention_probs.sum(dim=-1).mean():.4f} (应该接近1)")
    
    # 3. 测试 BERT 层
    print("\n3. 测试 BERT 编码器层")
    print("-" * 30)
    
    bert_layer = BERTLayer(hidden_size, num_attention_heads)
    layer_output, layer_attention = bert_layer(embedded_output, attention_mask, return_attention=True)
    
    print(f"编码器层输出形状: {layer_output.shape}")
    print(f"编码器层注意力形状: {layer_attention.shape}")
    
    # 4. 测试完整编码器
    print("\n4. 测试 BERT 编码器")
    print("-" * 30)
    
    encoder = BERTEncoder(hidden_size, num_hidden_layers, num_attention_heads)
    encoder_output, all_attentions = encoder(embedded_output, attention_mask, return_attention=True)
    
    print(f"编码器输出形状: {encoder_output.shape}")
    print(f"注意力层数: {len(all_attentions)}")
    print(f"每层注意力形状: {all_attentions[0].shape}")
    
    # 5. 测试池化层
    print("\n5. 测试 BERT 池化层")
    print("-" * 30)
    
    pooler = BERTPooler(hidden_size)
    pooled_output = pooler(encoder_output)
    
    print(f"池化输出形状: {pooled_output.shape}")
    print(f"池化输出统计: 均值={pooled_output.mean():.4f}, 标准差={pooled_output.std():.4f}")
    
    # 6. 参数统计
    print("\n6. 参数统计")
    print("-" * 30)
    
    total_params = 0
    for name, module in [
        ("嵌入层", embeddings),
        ("自注意力", attention),
        ("编码器层", bert_layer),
        ("完整编码器", encoder),
        ("池化层", pooler)
    ]:
        params = sum(p.numel() for p in module.parameters())
        total_params += params
        print(f"{name}: {params:,} 参数")
    
    print(f"总参数数量: {total_params:,}")
    
    print("\n✅ BERT 组件测试完成!")
    return True


def demonstrate_bert_attention():
    """演示 BERT 注意力机制"""
    print("\n" + "=" * 60)
    print("BERT 注意力机制演示")
    print("=" * 60)
    
    # 创建一个简单的例子
    vocab_size = 100
    hidden_size = 64  # 小一点便于演示
    num_heads = 4
    seq_len = 8
    batch_size = 1
    
    # 创建模型和数据
    attention = BERTSelfAttention(hidden_size, num_heads)
    input_embeddings = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    
    # 获取注意力权重
    attention.eval()
    with torch.no_grad():
        output, attention_weights = attention(input_embeddings, attention_mask, return_attention=True)
    
    print(f"输入形状: {input_embeddings.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 分析注意力模式
    avg_attention = attention_weights[0].mean(dim=0)  # 平均所有头
    
    print(f"\n注意力矩阵 (平均所有头):")
    print("每行表示该位置对所有位置的注意力分布")
    print("数值越大表示注意力越集中")
    
    for i in range(seq_len):
        attention_str = " ".join([f"{avg_attention[i, j]:.3f}" for j in range(seq_len)])
        print(f"位置 {i}: [{attention_str}]")
    
    # 找出最高注意力的位置对
    max_attention = avg_attention.max()
    max_pos = torch.where(avg_attention == max_attention)
    print(f"\n最高注意力: {max_attention:.4f}")
    print(f"位置: ({max_pos[0].item()}, {max_pos[1].item()})")
    
    return attention_weights


if __name__ == "__main__":
    # 测试所有组件
    test_bert_components()
    
    # 演示注意力机制
    demonstrate_bert_attention()
    
    print("\n" + "=" * 60)
    print("BERT 组件模块测试完成！")
    print("=" * 60)
