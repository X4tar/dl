"""
完整的 Transformer 模型实现
包含编码器和解码器的完整架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformer_components import (
    EncoderLayer, DecoderLayer, 
    create_padding_mask, create_look_ahead_mask
)
from positional_encoding import PositionalEncoding

class TransformerEncoder(nn.Module):
    """
    Transformer 编码器
    
    由多个编码器层堆叠而成，每个编码器层包含：
    - 多头自注意力机制
    - 位置前馈网络
    - 残差连接和层归一化
    """
    
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, 
                 max_seq_len, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # 编码器层堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        """
        前向传播
        
        Args:
            src: 源序列 [batch_size, src_seq_len]
            src_mask: 源序列掩码 [batch_size, 1, 1, src_seq_len]
            
        Returns:
            output: 编码器输出 [batch_size, src_seq_len, d_model]
            attention_weights: 各层的注意力权重
        """
        # 词嵌入 + 缩放
        x = self.embedding(src) * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # 存储注意力权重
        attention_weights = []
        
        # 通过所有编码器层
        for encoder_layer in self.encoder_layers:
            x, attn_weights = encoder_layer(x, src_mask)
            attention_weights.append(attn_weights)
            
        return x, attention_weights


class TransformerDecoder(nn.Module):
    """
    Transformer 解码器
    
    由多个解码器层堆叠而成，每个解码器层包含：
    - 掩码多头自注意力机制
    - 编码器-解码器注意力机制
    - 位置前馈网络
    - 残差连接和层归一化
    """
    
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, 
                 max_seq_len, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        self.d_model = d_model
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # 解码器层堆叠
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 输出投影层
        self.linear = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        前向传播
        
        Args:
            tgt: 目标序列 [batch_size, tgt_seq_len]
            encoder_output: 编码器输出 [batch_size, src_seq_len, d_model]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
            
        Returns:
            output: 解码器输出 [batch_size, tgt_seq_len, vocab_size]
            self_attention_weights: 自注意力权重
            cross_attention_weights: 交叉注意力权重
        """
        # 词嵌入 + 缩放
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # 存储注意力权重
        self_attention_weights = []
        cross_attention_weights = []
        
        # 通过所有解码器层
        for decoder_layer in self.decoder_layers:
            x, self_attn, cross_attn = decoder_layer(
                x, encoder_output, src_mask, tgt_mask
            )
            self_attention_weights.append(self_attn)
            cross_attention_weights.append(cross_attn)
        
        # 输出投影
        output = self.linear(x)
        
        return output, self_attention_weights, cross_attention_weights


class Transformer(nn.Module):
    """
    完整的 Transformer 模型
    
    包含编码器和解码器，可用于序列到序列的任务，如机器翻译。
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, 
                 n_layers=6, d_ff=2048, max_seq_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        
        # 编码器
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, n_heads, n_layers, 
            d_ff, max_seq_len, dropout
        )
        
        # 解码器
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, n_heads, n_layers, 
            d_ff, max_seq_len, dropout
        )
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        前向传播
        
        Args:
            src: 源序列 [batch_size, src_seq_len]
            tgt: 目标序列 [batch_size, tgt_seq_len]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
            
        Returns:
            output: 模型输出 [batch_size, tgt_seq_len, tgt_vocab_size]
            encoder_attention_weights: 编码器注意力权重
            decoder_self_attention_weights: 解码器自注意力权重
            decoder_cross_attention_weights: 解码器交叉注意力权重
        """
        # 编码器前向传播
        encoder_output, encoder_attention_weights = self.encoder(src, src_mask)
        
        # 解码器前向传播
        decoder_output, decoder_self_attention_weights, decoder_cross_attention_weights = \
            self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        return (decoder_output, encoder_attention_weights, 
                decoder_self_attention_weights, decoder_cross_attention_weights)
    
    def encode(self, src, src_mask=None):
        """
        仅编码（用于推理时）
        
        Args:
            src: 源序列 [batch_size, src_seq_len]
            src_mask: 源序列掩码
            
        Returns:
            encoder_output: 编码器输出
            attention_weights: 注意力权重
        """
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        仅解码（用于推理时）
        
        Args:
            tgt: 目标序列 [batch_size, tgt_seq_len]
            encoder_output: 编码器输出
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
            
        Returns:
            decoder_output: 解码器输出
            self_attention_weights: 自注意力权重
            cross_attention_weights: 交叉注意力权重
        """
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)


class TransformerForLanguageModeling(nn.Module):
    """
    用于语言建模的 Transformer（仅解码器）
    
    这是类似 GPT 的架构，用于自回归语言生成任务。
    """
    
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, 
                 d_ff=2048, max_seq_len=5000, dropout=0.1):
        super(TransformerForLanguageModeling, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # 解码器层（用作自回归层）
        self.decoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)  # 实际上是自注意力层
            for _ in range(n_layers)
        ])
        
        # 输出层
        self.ln_f = nn.LayerNorm(d_model)  # 最终层归一化
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重共享（可选）
        self.lm_head.weight = self.embedding.weight
        
    def forward(self, input_ids, attention_mask=None):
        """
        前向传播
        
        Args:
            input_ids: 输入序列 [batch_size, seq_len]
            attention_mask: 注意力掩码
            
        Returns:
            logits: 输出logits [batch_size, seq_len, vocab_size]
            attention_weights: 注意力权重列表
        """
        batch_size, seq_len = input_ids.size()
        
        # 创建因果掩码（下三角矩阵）
        causal_mask = create_look_ahead_mask(seq_len).to(input_ids.device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        causal_mask = causal_mask.repeat(batch_size, 1, 1, 1)  # [batch_size, 1, seq_len, seq_len]
        
        # 如果提供了注意力掩码，结合因果掩码
        if attention_mask is not None:
            # attention_mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask * attention_mask
        
        # 词嵌入 + 缩放
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # 存储注意力权重
        attention_weights = []
        
        # 通过所有层
        for layer in self.decoder_layers:
            x, attn_weights = layer(x, causal_mask)
            attention_weights.append(attn_weights)
        
        # 最终层归一化
        x = self.ln_f(x)
        
        # 输出投影
        logits = self.lm_head(x)
        
        return logits, attention_weights
    
    def generate(self, input_ids, max_length=50, temperature=1.0, do_sample=True):
        """
        文本生成
        
        Args:
            input_ids: 初始输入序列 [batch_size, seq_len]
            max_length: 最大生成长度
            temperature: 采样温度
            do_sample: 是否使用采样
            
        Returns:
            generated_ids: 生成的序列 [batch_size, max_length]
        """
        self.eval()
        batch_size = input_ids.size(0)
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # 前向传播
                logits, _ = self.forward(generated)
                
                # 获取最后一个位置的logits
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    # 采样下一个token
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # 贪心选择
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # 添加到生成序列
                generated = torch.cat([generated, next_token], dim=1)
                
                # 检查是否生成了结束符（假设词汇表中有结束符）
                # 这里可以根据实际需求添加停止条件
                
        return generated


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model):
    """
    初始化模型权重
    使用Xavier初始化
    """
    for name, param in model.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)


if __name__ == "__main__":
    print("测试 Transformer 模型...\n")
    
    # 模型参数
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    max_seq_len = 100
    
    # 测试完整的 Transformer
    print("1. 测试完整的 Transformer（编码器-解码器）")
    transformer = Transformer(
        src_vocab_size, tgt_vocab_size, d_model, n_heads, 
        n_layers, d_ff, max_seq_len
    )
    
    print(f"模型参数数量: {count_parameters(transformer):,}")
    
    # 创建测试数据
    batch_size = 2
    src_seq_len = 20
    tgt_seq_len = 15
    
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    print(f"源序列形状: {src.shape}")
    print(f"目标序列形状: {tgt.shape}")
    
    # 创建掩码
    src_mask = create_padding_mask(src)
    tgt_mask = create_look_ahead_mask(tgt_seq_len).unsqueeze(0).unsqueeze(0)
    
    # 前向传播
    output, enc_attn, dec_self_attn, dec_cross_attn = transformer(src, tgt, src_mask, tgt_mask)
    
    print(f"输出形状: {output.shape}")
    print(f"编码器注意力层数: {len(enc_attn)}")
    print(f"解码器自注意力层数: {len(dec_self_attn)}")
    print(f"解码器交叉注意力层数: {len(dec_cross_attn)}")
    
    # 测试语言建模 Transformer
    print("\n2. 测试语言建模 Transformer（仅解码器）")
    lm_transformer = TransformerForLanguageModeling(
        src_vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len
    )
    
    print(f"语言模型参数数量: {count_parameters(lm_transformer):,}")
    
    # 测试语言建模
    input_ids = torch.randint(0, src_vocab_size, (batch_size, 20))
    logits, lm_attention = lm_transformer(input_ids)
    
    print(f"语言模型输入形状: {input_ids.shape}")
    print(f"语言模型输出形状: {logits.shape}")
    print(f"语言模型注意力层数: {len(lm_attention)}")
    
    # 测试文本生成
    print("\n3. 测试文本生成")
    generated = lm_transformer.generate(input_ids[:1, :5], max_length=20)
    print(f"生成序列形状: {generated.shape}")
    print(f"原始输入: {input_ids[0, :5].tolist()}")
    print(f"生成序列: {generated[0].tolist()}")
    
    print("\n所有测试完成！")
