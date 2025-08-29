"""
位置编码实现
详细解释和实现 Transformer 中的位置编码机制
"""

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

class PositionalEncoding(nn.Module):
    """
    位置编码
    
    由于 Transformer 没有递归或卷积，它无法捕获序列中的位置信息。
    位置编码为每个位置添加唯一的编码，让模型能够理解词语的顺序。
    
    公式：
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    其中：
    - pos: 位置索引
    - i: 维度索引
    - d_model: 模型维度
    """
    
    pe: torch.Tensor  # 声明 pe 为 buffer 属性
    
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵 [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)
        
        # 创建位置索引 [max_seq_len, 1]
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        # 创建除数项 [d_model//2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置用cos
        
        # 添加批次维度 [1, max_seq_len, d_model]
        pe = pe.unsqueeze(0)
        
        self.pe: torch.Tensor
        # 注册为buffer，不参与梯度更新
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入嵌入 [batch_size, seq_len, d_model]
            
        Returns:
            output: 添加位置编码后的输出 [batch_size, seq_len, d_model]
        """
        # 添加位置编码到输入嵌入
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    可学习的位置编码
    
    与固定的正弦余弦位置编码不同，这种位置编码的参数可以通过训练学习。
    """
    
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # 创建可学习的位置嵌入
        self.pe = nn.Embedding(max_seq_len, d_model)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入嵌入 [batch_size, seq_len, d_model]
            
        Returns:
            output: 添加位置编码后的输出 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()
        
        # 创建位置索引
        positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        
        # 获取位置编码
        position_encodings = self.pe(positions)
        
        # 添加到输入
        x = x + position_encodings
        return self.dropout(x)


def visualize_positional_encoding(d_model=512, max_seq_len=100):
    """
    可视化位置编码
    
    Args:
        d_model: 模型维度
        max_seq_len: 最大序列长度
    """
    # 创建位置编码
    pe = PositionalEncoding(d_model, max_seq_len)
    
    # 获取位置编码矩阵
    encoding = pe.pe.squeeze(0).detach().numpy()  # [max_seq_len, d_model]
    
    # 创建图形
    plt.figure(figsize=(15, 8))
    
    # 第一个子图：完整的位置编码热力图
    plt.subplot(2, 2, 1)
    plt.imshow(encoding.T, cmap='RdYlBu', aspect='auto')
    plt.title('完整位置编码热力图')
    plt.xlabel('位置')
    plt.ylabel('维度')
    plt.colorbar()
    
    # 第二个子图：前几个位置的编码值
    plt.subplot(2, 2, 2)
    for pos in range(min(10, max_seq_len)):
        plt.plot(encoding[pos, :50], label=f'位置 {pos}')
    plt.title('前10个位置的编码值（前50维）')
    plt.xlabel('维度')
    plt.ylabel('编码值')
    plt.legend()
    plt.grid(True)
    
    # 第三个子图：特定维度在不同位置的编码值
    plt.subplot(2, 2, 3)
    dims_to_plot = [0, 1, 10, 11, 50, 51]
    for dim in dims_to_plot:
        plt.plot(encoding[:max_seq_len, dim], label=f'维度 {dim}')
    plt.title('不同维度在各位置的编码值')
    plt.xlabel('位置')
    plt.ylabel('编码值')
    plt.legend()
    plt.grid(True)
    
    # 第四个子图：正弦和余弦函数的波长
    plt.subplot(2, 2, 4)
    # 显示不同频率的正弦波
    x = np.arange(max_seq_len)
    frequencies = [1, 2, 4, 8]
    for freq in frequencies:
        y = np.sin(x / (10000 ** (freq / d_model)))
        plt.plot(x, y, label=f'频率项 {freq}')
    plt.title('不同频率的正弦波')
    plt.xlabel('位置')
    plt.ylabel('值')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('transfomer/positional_encoding_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_positional_encoding_properties():
    """
    分析位置编码的特性
    """
    print("分析位置编码的特性...")
    
    d_model = 512
    max_seq_len = 100
    
    # 创建位置编码
    pe = PositionalEncoding(d_model, max_seq_len)
    encoding = pe.pe.squeeze(0)  # [max_seq_len, d_model]
    
    print(f"位置编码形状: {encoding.shape}")
    print(f"编码值范围: [{encoding.min():.4f}, {encoding.max():.4f}]")
    
    # 分析不同位置编码的相似性
    print("\n位置编码相似性分析:")
    positions_to_compare = [0, 1, 10, 50]
    
    for i, pos1 in enumerate(positions_to_compare):
        for pos2 in positions_to_compare[i+1:]:
            similarity = torch.cosine_similarity(
                encoding[pos1].unsqueeze(0), 
                encoding[pos2].unsqueeze(0)
            )
            print(f"位置 {pos1} 和位置 {pos2} 的余弦相似度: {similarity.item():.4f}")
    
    # 分析相对位置关系
    print("\n相对位置关系分析:")
    base_pos = 10
    relative_distances = [1, 5, 10, 20]
    
    for dist in relative_distances:
        if base_pos + dist < max_seq_len:
            similarity = torch.cosine_similarity(
                encoding[base_pos].unsqueeze(0),
                encoding[base_pos + dist].unsqueeze(0)
            )
            print(f"位置 {base_pos} 和位置 {base_pos + dist} (距离 {dist}) 的相似度: {similarity.item():.4f}")


def compare_encoding_methods():
    """
    比较不同位置编码方法
    """
    print("比较不同位置编码方法...")
    
    d_model = 256
    seq_len = 50
    batch_size = 1
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 固定位置编码
    fixed_pe = PositionalEncoding(d_model, seq_len * 2)
    x_fixed = fixed_pe(x.clone())
    
    # 可学习位置编码
    learnable_pe = LearnablePositionalEncoding(d_model, seq_len * 2)
    x_learnable = learnable_pe(x.clone())
    
    print(f"原始输入形状: {x.shape}")
    print(f"固定编码后形状: {x_fixed.shape}")
    print(f"可学习编码后形状: {x_learnable.shape}")
    
    print(f"\n固定编码后的值范围: [{x_fixed.min():.4f}, {x_fixed.max():.4f}]")
    print(f"可学习编码后的值范围: [{x_learnable.min():.4f}, {x_learnable.max():.4f}]")


def demonstrate_position_encoding_importance():
    """
    演示位置编码的重要性
    """
    print("演示位置编码的重要性...")
    
    # 创建一个简单的例子
    d_model = 8
    seq_len = 5
    
    # 创建相同的词嵌入但在不同位置
    word_embedding = torch.ones(1, 1, d_model)  # 相同的词
    
    # 不同位置的相同词
    sentence = word_embedding.repeat(1, seq_len, 1)  # [1, 5, 8]
    
    print("原始句子（相同的词在不同位置）:")
    print(f"形状: {sentence.shape}")
    print("前两个位置的嵌入是否相同:", torch.equal(sentence[0, 0], sentence[0, 1]))
    
    # 添加位置编码
    pe = PositionalEncoding(d_model, seq_len)
    sentence_with_pe = pe(sentence)
    
    print("\n添加位置编码后:")
    print("前两个位置的嵌入是否相同:", torch.equal(sentence_with_pe[0, 0], sentence_with_pe[0, 1]))
    print("位置0的编码:", sentence_with_pe[0, 0].detach().numpy()[:4])
    print("位置1的编码:", sentence_with_pe[0, 1].detach().numpy()[:4])
    print("位置2的编码:", sentence_with_pe[0, 2].detach().numpy()[:4])


if __name__ == "__main__":
    print("位置编码详细分析\n" + "="*50)
    
    # 演示位置编码的重要性
    demonstrate_position_encoding_importance()
    
    print("\n" + "="*50)
    
    # 分析位置编码特性
    analyze_positional_encoding_properties()
    
    print("\n" + "="*50)
    
    # 比较不同编码方法
    compare_encoding_methods()
    
    print("\n" + "="*50)
    
    # 可视化位置编码（需要matplotlib）
    try:
        print("生成位置编码可视化...")
        visualize_positional_encoding(d_model=128, max_seq_len=50)
        print("可视化图片已保存到 'transfomer/positional_encoding_visualization.png'")
    except ImportError:
        print("需要安装 matplotlib 来生成可视化")
    except Exception as e:
        print(f"生成可视化时出错: {e}")
    
    print("\n位置编码分析完成！")
