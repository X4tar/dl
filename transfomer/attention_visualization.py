"""
注意力可视化工具
帮助理解和可视化 Transformer 中的注意力机制
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_model import Transformer, TransformerForLanguageModeling
from transformer_components import create_padding_mask, create_look_ahead_mask
from train_transformer import build_vocab, create_sample_data

class AttentionVisualizer:
    """注意力可视化器"""
    
    def __init__(self, model, src_vocab, tgt_vocab, src_id2word, tgt_id2word, device):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_id2word = src_id2word
        self.tgt_id2word = tgt_id2word
        self.device = device
        self.model.eval()
    
    def prepare_input(self, sentence, vocab, is_target=False):
        """准备输入序列"""
        words = sentence.lower().split()
        token_ids = [vocab.get(word, 3) for word in words]  # 3是UNK标记
        
        if is_target:
            token_ids = [1] + token_ids  # 添加BOS标记
        
        return torch.tensor([token_ids], dtype=torch.long).to(self.device), words
    
    def get_attention_weights(self, src_sentence, tgt_sentence=None):
        """
        获取注意力权重
        
        Args:
            src_sentence: 源句子
            tgt_sentence: 目标句子（如果是seq2seq模型）
            
        Returns:
            attention_data: 包含各种注意力权重的字典
        """
        # 准备输入
        src_tensor, src_words = self.prepare_input(src_sentence, self.src_vocab)
        
        if isinstance(self.model, Transformer):
            # 编码器-解码器模型
            if tgt_sentence is None:
                raise ValueError("编码器-解码器模型需要目标句子")
            
            tgt_tensor, tgt_words = self.prepare_input(tgt_sentence, self.tgt_vocab, is_target=True)
            
            # 创建掩码
            src_mask = create_padding_mask(src_tensor)
            tgt_seq_len = tgt_tensor.size(1)
            tgt_mask = create_look_ahead_mask(tgt_seq_len).to(self.device)
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
            
            # 前向传播
            with torch.no_grad():
                output, enc_attn, dec_self_attn, dec_cross_attn = self.model(
                    src_tensor, tgt_tensor, src_mask, tgt_mask
                )
            
            return {
                'src_words': src_words,
                'tgt_words': ['<BOS>'] + tgt_words,
                'encoder_attention': enc_attn,
                'decoder_self_attention': dec_self_attn,
                'decoder_cross_attention': dec_cross_attn,
                'output': output
            }
        
        else:
            # 仅解码器模型（语言模型）
            with torch.no_grad():
                logits, attention_weights = self.model(src_tensor)
            
            return {
                'words': src_words,
                'attention_weights': attention_weights,
                'output': logits
            }
    
    def visualize_encoder_attention(self, attention_weights, words, layer_idx=0, head_idx=0, 
                                  save_path=None):
        """
        可视化编码器自注意力
        
        Args:
            attention_weights: 注意力权重 [batch, heads, seq_len, seq_len]
            words: 词列表
            layer_idx: 层索引
            head_idx: 头索引
            save_path: 保存路径
        """
        # 获取指定层和头的注意力权重
        attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()
        seq_len = len(words)
        attn = attn[:seq_len, :seq_len]
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 绘制热力图
        sns.heatmap(attn, 
                   xticklabels=words, 
                   yticklabels=words,
                   annot=True, 
                   fmt='.3f',
                   cmap='Blues',
                   cbar_kws={'label': '注意力权重'})
        
        plt.title(f'编码器自注意力 - 第{layer_idx+1}层, 第{head_idx+1}头')
        plt.xlabel('键 (Key)')
        plt.ylabel('查询 (Query)')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_cross_attention(self, attention_weights, src_words, tgt_words, 
                                 layer_idx=0, head_idx=0, save_path=None):
        """
        可视化编码器-解码器交叉注意力
        
        Args:
            attention_weights: 交叉注意力权重
            src_words: 源词列表
            tgt_words: 目标词列表
            layer_idx: 层索引
            head_idx: 头索引
            save_path: 保存路径
        """
        # 获取指定层和头的注意力权重
        attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()
        src_len = len(src_words)
        tgt_len = len(tgt_words)
        attn = attn[:tgt_len, :src_len]
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 绘制热力图
        sns.heatmap(attn,
                   xticklabels=src_words,
                   yticklabels=tgt_words,
                   annot=True,
                   fmt='.3f',
                   cmap='Reds',
                   cbar_kws={'label': '注意力权重'})
        
        plt.title(f'编码器-解码器交叉注意力 - 第{layer_idx+1}层, 第{head_idx+1}头')
        plt.xlabel('源序列 (编码器输出)')
        plt.ylabel('目标序列 (解码器输入)')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_multi_head_attention(self, attention_weights, words, layer_idx=0, 
                                     max_heads=8, save_path=None):
        """
        可视化多头注意力
        
        Args:
            attention_weights: 注意力权重
            words: 词列表
            layer_idx: 层索引
            max_heads: 最大显示头数
            save_path: 保存路径
        """
        # 获取指定层的注意力权重
        attn_layer = attention_weights[layer_idx][0].cpu().numpy()  # [heads, seq_len, seq_len]
        seq_len = len(words)
        n_heads = min(attn_layer.shape[0], max_heads)
        
        # 计算子图布局
        cols = min(4, n_heads)
        rows = (n_heads + cols - 1) // cols
        
        # 创建图形
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for head_idx in range(n_heads):
            row = head_idx // cols
            col = head_idx % cols
            ax = axes[row, col]
            
            # 获取当前头的注意力权重
            attn_head = attn_layer[head_idx, :seq_len, :seq_len]
            
            # 绘制热力图
            sns.heatmap(attn_head,
                       xticklabels=words,
                       yticklabels=words,
                       annot=False,
                       cmap='Blues',
                       ax=ax,
                       cbar=False)
            
            ax.set_title(f'头 {head_idx+1}')
            ax.set_xlabel('键')
            ax.set_ylabel('查询')
            
            # 调整标签
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', rotation=0, labelsize=8)
        
        # 隐藏多余的子图
        for idx in range(n_heads, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(f'多头注意力 - 第{layer_idx+1}层', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_attention_patterns(self, src_sentence, tgt_sentence=None, save_dir=None):
        """
        可视化完整的注意力模式
        
        Args:
            src_sentence: 源句子
            tgt_sentence: 目标句子
            save_dir: 保存目录
        """
        print(f"可视化注意力模式...")
        print(f"源句子: {src_sentence}")
        if tgt_sentence:
            print(f"目标句子: {tgt_sentence}")
        
        # 获取注意力权重
        attention_data = self.get_attention_weights(src_sentence, tgt_sentence)
        
        if isinstance(self.model, Transformer):
            # 编码器-解码器模型
            src_words = attention_data['src_words']
            tgt_words = attention_data['tgt_words']
            
            # 可视化编码器自注意力
            print("\n1. 编码器自注意力")
            for layer_idx in range(min(2, len(attention_data['encoder_attention']))):
                self.visualize_encoder_attention(
                    attention_data['encoder_attention'], 
                    src_words, 
                    layer_idx=layer_idx,
                    save_path=f"{save_dir}/encoder_attention_layer_{layer_idx}.png" if save_dir else None
                )
            
            # 可视化多头注意力
            print("\n2. 编码器多头注意力")
            self.visualize_multi_head_attention(
                attention_data['encoder_attention'],
                src_words,
                layer_idx=0,
                save_path=f"{save_dir}/encoder_multihead_attention.png" if save_dir else None
            )
            
            # 可视化交叉注意力
            print("\n3. 编码器-解码器交叉注意力")
            for layer_idx in range(min(2, len(attention_data['decoder_cross_attention']))):
                self.visualize_cross_attention(
                    attention_data['decoder_cross_attention'],
                    src_words,
                    tgt_words,
                    layer_idx=layer_idx,
                    save_path=f"{save_dir}/cross_attention_layer_{layer_idx}.png" if save_dir else None
                )
        
        else:
            # 仅解码器模型
            words = attention_data['words']
            
            # 可视化自注意力
            print("\n1. 自注意力")
            for layer_idx in range(min(2, len(attention_data['attention_weights']))):
                self.visualize_encoder_attention(
                    attention_data['attention_weights'],
                    words,
                    layer_idx=layer_idx,
                    save_path=f"{save_dir}/self_attention_layer_{layer_idx}.png" if save_dir else None
                )
            
            # 可视化多头注意力
            print("\n2. 多头注意力")
            self.visualize_multi_head_attention(
                attention_data['attention_weights'],
                words,
                layer_idx=0,
                save_path=f"{save_dir}/multihead_attention.png" if save_dir else None
            )


def analyze_attention_patterns():
    """分析注意力模式"""
    print("分析注意力模式...")
    
    # 创建样本数据
    english_sentences, french_sentences = create_sample_data()
    
    # 构建词汇表
    src_vocab, src_id2word = build_vocab(english_sentences, min_freq=1)
    tgt_vocab, tgt_id2word = build_vocab(french_sentences, min_freq=1)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建一个小模型用于演示
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=50,
        dropout=0.1
    ).to(device)
    
    # 创建可视化器
    visualizer = AttentionVisualizer(
        model, src_vocab, tgt_vocab, src_id2word, tgt_id2word, device
    )
    
    # 测试句子
    test_sentences = [
        ("hello world", "bonjour monde"),
        ("how are you", "comment allez vous"),
        ("good morning", "bonjour matin")
    ]
    
    for i, (src, tgt) in enumerate(test_sentences):
        print(f"\n{'='*50}")
        print(f"示例 {i+1}")
        
        try:
            visualizer.visualize_attention_patterns(src, tgt, save_dir="transfomer")
        except Exception as e:
            print(f"可视化时出错: {e}")
            print("这可能是因为模型未训练，注意力权重是随机的")


def demonstrate_attention_mechanics():
    """演示注意力机制的工作原理"""
    print("演示注意力机制的工作原理...\n")
    
    # 创建简单的注意力计算示例
    seq_len = 5
    d_model = 8
    
    # 模拟输入序列
    print("1. 输入序列")
    X = torch.randn(1, seq_len, d_model)
    print(f"输入形状: {X.shape}")
    print(f"输入内容:\n{X[0, :, :4]}")  # 只显示前4维
    
    # 线性变换得到Q, K, V
    print("\n2. 计算 Query, Key, Value")
    W_q = torch.randn(d_model, d_model)
    W_k = torch.randn(d_model, d_model)
    W_v = torch.randn(d_model, d_model)
    
    Q = torch.matmul(X, W_q)
    K = torch.matmul(X, W_k)
    V = torch.matmul(X, W_v)
    
    print(f"Q 形状: {Q.shape}")
    print(f"K 形状: {K.shape}")
    print(f"V 形状: {V.shape}")
    
    # 计算注意力分数
    print("\n3. 计算注意力分数")
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_model)
    print(f"注意力分数形状: {scores.shape}")
    print(f"注意力分数:\n{scores[0]}")
    
    # 应用 softmax
    print("\n4. 应用 Softmax")
    attention_weights = torch.softmax(scores, dim=-1)
    print(f"注意力权重:\n{attention_weights[0]}")
    print(f"每行权重之和: {attention_weights[0].sum(dim=-1)}")
    
    # 计算输出
    print("\n5. 计算输出")
    output = torch.matmul(attention_weights, V)
    print(f"输出形状: {output.shape}")
    print(f"输出内容:\n{output[0, :, :4]}")  # 只显示前4维
    
    # 可视化注意力权重
    print("\n6. 可视化注意力权重")
    plt.figure(figsize=(8, 6))
    sns.heatmap(attention_weights[0].numpy(), 
               annot=True, 
               fmt='.3f',
               cmap='Blues',
               xticklabels=[f'位置{i}' for i in range(seq_len)],
               yticklabels=[f'位置{i}' for i in range(seq_len)])
    plt.title('注意力权重矩阵')
    plt.xlabel('Key 位置')
    plt.ylabel('Query 位置')
    plt.savefig('transfomer/attention_demo.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Transformer 注意力可视化")
    print("=" * 60)
    
    # 演示注意力机制
    try:
        demonstrate_attention_mechanics()
    except ImportError:
        print("需要安装 matplotlib 和 seaborn 来运行可视化")
    except Exception as e:
        print(f"演示时出错: {e}")
    
    print("\n" + "=" * 60)
    
    # 分析注意力模式
    try:
        analyze_attention_patterns()
    except Exception as e:
        print(f"分析注意力模式时出错: {e}")
        print("这可能是因为缺少依赖或模型配置问题")
    
    print("\n注意力可视化完成！")
