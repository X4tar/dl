"""
图像分块和嵌入模块
实现 ViT 中的图像预处理：分块、展平、线性投影和位置编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class PatchEmbedding(nn.Module):
    """
    图像分块嵌入层
    将输入图像分割成 patches 并转换为嵌入向量
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # 计算 patch 数量
        self.n_patches = (img_size // patch_size) ** 2
        
        # 使用卷积来实现分块和线性投影
        # 相当于将每个 patch 展平后通过线性层
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像 [batch_size, channels, height, width]
            
        Returns:
            patches: 图像块嵌入 [batch_size, n_patches, embed_dim]
        """
        batch_size, channels, height, width = x.shape
        
        # 检查输入尺寸
        assert height == self.img_size and width == self.img_size, \
            f"输入图像尺寸 ({height}, {width}) 与期望尺寸 ({self.img_size}, {self.img_size}) 不匹配"
        
        # 应用卷积投影: [B, C, H, W] -> [B, embed_dim, H/P, W/P]
        x = self.projection(x)
        
        # 展平并转置: [B, embed_dim, H/P, W/P] -> [B, n_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        
        return x


class PositionalEncoding(nn.Module):
    """
    位置编码
    为每个 patch 添加位置信息
    """
    
    def __init__(self, n_patches, embed_dim, dropout=0.1):
        super().__init__()
        self.n_patches = n_patches
        self.embed_dim = embed_dim
        
        # 可学习的位置嵌入
        self.position_embeddings = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        添加位置编码
        
        Args:
            x: patch 嵌入 [batch_size, n_patches + 1, embed_dim] (包含 CLS token)
            
        Returns:
            x: 添加位置编码后的嵌入 [batch_size, n_patches + 1, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # 添加位置编码
        x = x + self.position_embeddings[:, :seq_len, :]
        
        return self.dropout(x)


class ViTEmbedding(nn.Module):
    """
    完整的 ViT 嵌入层
    包含图像分块、CLS token、位置编码
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, dropout=0.1):
        super().__init__()
        
        # 图像分块嵌入
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 位置编码
        n_patches = self.patch_embedding.n_patches
        self.position_encoding = PositionalEncoding(n_patches, embed_dim, dropout)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像 [batch_size, channels, height, width]
            
        Returns:
            embeddings: 完整嵌入 [batch_size, n_patches + 1, embed_dim]
        """
        batch_size = x.shape[0]
        
        # 1. 图像分块嵌入
        patch_embeddings = self.patch_embedding(x)  # [B, n_patches, embed_dim]
        
        # 2. 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, embed_dim]
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)  # [B, n_patches + 1, embed_dim]
        
        # 3. 添加位置编码
        embeddings = self.position_encoding(embeddings)
        
        return embeddings


def visualize_patches(image, patch_size=16):
    """
    可视化图像分块过程
    
    Args:
        image: PIL Image 或 numpy array
        patch_size: patch 大小
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    height, width = image.shape[:2]
    n_patches_h = height // patch_size
    n_patches_w = width // patch_size
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 原始图像
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    # 带网格的图像
    axes[0, 1].imshow(image)
    for i in range(0, height, patch_size):
        axes[0, 1].axhline(y=i, color='red', linewidth=1)
    for j in range(0, width, patch_size):
        axes[0, 1].axvline(x=j, color='red', linewidth=1)
    axes[0, 1].set_title(f'分块网格 ({patch_size}x{patch_size})')
    axes[0, 1].axis('off')
    
    # 随机选择一些 patches 显示
    n_display = min(16, n_patches_h * n_patches_w)
    patches_to_show = np.random.choice(n_patches_h * n_patches_w, n_display, replace=False)
    
    # 提取 patches
    patches = []
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            patch_idx = i * n_patches_w + j
            if patch_idx in patches_to_show:
                patch = image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                patches.append(patch)
    
    # 显示部分 patches
    rows = int(np.sqrt(len(patches)))
    cols = (len(patches) + rows - 1) // rows
    
    for idx, patch in enumerate(patches[:16]):
        row = idx // 4
        col = idx % 4
        if row < 4 and col < 4:
            axes[0, 1].add_patch(Rectangle((col*patch_size*width//64, row*patch_size*height//64), 
                                          patch_size*width//64, patch_size*height//64, 
                                          fill=False, edgecolor='blue', linewidth=2))
    
    # 显示选中的 patches
    fig2, axes2 = plt.subplots(4, 4, figsize=(10, 10))
    for idx, patch in enumerate(patches[:16]):
        row = idx // 4
        col = idx % 4
        axes2[row, col].imshow(patch)
        axes2[row, col].set_title(f'Patch {idx+1}')
        axes2[row, col].axis('off')
    
    # 隐藏多余的子图
    for idx in range(len(patches), 16):
        row = idx // 4
        col = idx % 4
        axes2[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('vit/patch_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return patches


def demonstrate_embedding_process():
    """演示完整的嵌入过程"""
    print("=" * 60)
    print("演示 ViT 嵌入过程")
    print("=" * 60)
    
    # 参数设置
    img_size = 224
    patch_size = 16
    in_channels = 3
    embed_dim = 768
    batch_size = 2
    
    # 创建随机图像
    images = torch.randn(batch_size, in_channels, img_size, img_size)
    print(f"输入图像形状: {images.shape}")
    
    # 1. 测试图像分块嵌入
    print("\n1. 图像分块嵌入")
    print("-" * 30)
    
    patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
    patch_embeddings = patch_embed(images)
    
    n_patches = (img_size // patch_size) ** 2
    print(f"Patch 大小: {patch_size}x{patch_size}")
    print(f"Patch 数量: {n_patches}")
    print(f"Patch 嵌入形状: {patch_embeddings.shape}")
    
    # 2. 测试位置编码
    print("\n2. 位置编码")
    print("-" * 30)
    
    pos_encoding = PositionalEncoding(n_patches, embed_dim)
    
    # 添加 CLS token 以匹配位置编码
    cls_token = torch.randn(batch_size, 1, embed_dim)
    embeddings_with_cls = torch.cat([cls_token, patch_embeddings], dim=1)
    
    embeddings_with_pos = pos_encoding(embeddings_with_cls)
    print(f"位置编码形状: {pos_encoding.position_embeddings.shape}")
    print(f"添加位置编码后形状: {embeddings_with_pos.shape}")
    
    # 3. 测试完整嵌入
    print("\n3. 完整 ViT 嵌入")
    print("-" * 30)
    
    vit_embedding = ViTEmbedding(img_size, patch_size, in_channels, embed_dim)
    final_embeddings = vit_embedding(images)
    
    print(f"最终嵌入形状: {final_embeddings.shape}")
    print(f"序列长度: {final_embeddings.shape[1]} (= {n_patches} patches + 1 CLS token)")
    
    # 4. 分析嵌入特性
    print("\n4. 嵌入特性分析")
    print("-" * 30)
    
    # CLS token 统计
    cls_embeddings = final_embeddings[:, 0, :]  # [batch_size, embed_dim]
    print(f"CLS token 统计:")
    print(f"  均值: {cls_embeddings.mean().item():.4f}")
    print(f"  标准差: {cls_embeddings.std().item():.4f}")
    
    # Patch embeddings 统计
    patch_embeddings = final_embeddings[:, 1:, :]  # [batch_size, n_patches, embed_dim]
    print(f"Patch embeddings 统计:")
    print(f"  均值: {patch_embeddings.mean().item():.4f}")
    print(f"  标准差: {patch_embeddings.std().item():.4f}")
    
    # 位置编码影响
    embeddings_no_pos = embeddings_with_cls
    pos_effect = torch.abs(embeddings_with_pos - embeddings_no_pos).mean()
    print(f"位置编码影响大小: {pos_effect.item():.4f}")
    
    # 5. 参数数量统计
    print("\n5. 参数数量统计")
    print("-" * 30)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    patch_params = count_parameters(patch_embed)
    pos_params = count_parameters(pos_encoding)
    total_params = count_parameters(vit_embedding)
    
    print(f"图像分块嵌入参数: {patch_params:,}")
    print(f"位置编码参数: {pos_params:,}")
    print(f"总参数数量: {total_params:,}")
    
    return final_embeddings


def analyze_patch_patterns():
    """分析不同 patch 大小的影响"""
    print("\n" + "=" * 60)
    print("分析不同 Patch 大小的影响")
    print("=" * 60)
    
    img_size = 224
    patch_sizes = [8, 16, 32, 56]
    embed_dim = 768
    
    print(f"图像尺寸: {img_size}x{img_size}")
    print(f"测试的 Patch 大小: {patch_sizes}")
    
    # 创建测试图像
    test_image = torch.randn(1, 3, img_size, img_size)
    
    results = []
    
    for patch_size in patch_sizes:
        if img_size % patch_size != 0:
            print(f"跳过 patch_size={patch_size} (无法整除图像尺寸)")
            continue
            
        # 创建嵌入层
        embedding = ViTEmbedding(img_size, patch_size, 3, embed_dim)
        
        # 计算嵌入
        embeddings = embedding(test_image)
        
        # 统计信息
        n_patches = (img_size // patch_size) ** 2
        seq_len = embeddings.shape[1]
        
        # 参数数量
        params = sum(p.numel() for p in embedding.parameters() if p.requires_grad)
        
        results.append({
            'patch_size': patch_size,
            'n_patches': n_patches,
            'seq_len': seq_len,
            'params': params,
            'embeddings_shape': embeddings.shape
        })
        
        print(f"\nPatch 大小 {patch_size}x{patch_size}:")
        print(f"  Patch 数量: {n_patches}")
        print(f"  序列长度: {seq_len}")
        print(f"  参数数量: {params:,}")
        print(f"  嵌入形状: {embeddings.shape}")
    
    # 比较分析
    print(f"\n比较分析:")
    print("-" * 40)
    print(f"{'Patch大小':<8} {'Patch数量':<8} {'序列长度':<8} {'参数数量':<10}")
    print("-" * 40)
    
    for result in results:
        print(f"{result['patch_size']:<8} {result['n_patches']:<8} "
              f"{result['seq_len']:<8} {result['params']:<10,}")
    
    print(f"\n观察:")
    print("- 更小的 patch 产生更长的序列，计算复杂度更高")
    print("- 更大的 patch 丢失更多细节信息")
    print("- patch_size=16 是常见的平衡选择")


if __name__ == "__main__":
    # 演示嵌入过程
    embeddings = demonstrate_embedding_process()
    
    # 分析不同 patch 大小
    analyze_patch_patterns()
    
    # 创建示例图像进行可视化（如果可能）
    try:
        print("\n" + "=" * 60)
        print("创建示例图像可视化")
        print("=" * 60)
        
        # 创建彩色测试图像
        img_array = np.random.rand(224, 224, 3)
        # 添加一些结构
        img_array[50:150, 50:150] = [1, 0, 0]  # 红色方块
        img_array[100:200, 100:200] = [0, 1, 0]  # 绿色方块
        
        print("生成了示例图像用于可视化")
        print("运行 visualize_patches() 查看分块效果")
        
    except Exception as e:
        print(f"图像可视化需要 PIL 和 matplotlib: {e}")
    
    print("\n" + "=" * 60)
    print("ViT 图像嵌入模块测试完成！")
    print("=" * 60)
