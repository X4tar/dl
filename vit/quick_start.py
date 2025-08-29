"""
Vision Transformer (ViT) 快速入门
演示 ViT 的基本功能和使用方法
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 确保能导入本地模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vit_components import test_components, demonstrate_attention_mechanism
from patch_embedding import demonstrate_embedding_process, analyze_patch_patterns
from vit_model import create_vit_model, count_parameters, test_vit_models

def quick_demo():
    """快速演示 ViT 的核心功能"""
    print("🚀 Vision Transformer (ViT) 快速入门")
    print("=" * 60)
    
    # 1. 创建一个简单的 ViT 模型
    print("\n1. 创建 ViT 模型")
    print("-" * 30)
    
    model = create_vit_model(
        model_name='vit_tiny',
        img_size=224,
        patch_size=16,
        num_classes=10
    )
    
    total_params, trainable_params = count_parameters(model)
    print(f"✓ 创建了 ViT-Tiny 模型")
    print(f"  - 图像尺寸: 224x224")
    print(f"  - Patch 大小: 16x16")
    print(f"  - 分类类别: 10")
    print(f"  - 参数数量: {total_params:,}")
    
    # 2. 创建随机输入图像
    print("\n2. 创建输入图像")
    print("-" * 30)
    
    batch_size = 2
    channels = 3
    height = width = 224
    
    # 创建随机图像
    images = torch.randn(batch_size, channels, height, width)
    print(f"✓ 创建了随机图像: {images.shape}")
    
    # 3. 模型推理
    print("\n3. 模型推理")
    print("-" * 30)
    
    model.eval()
    with torch.no_grad():
        # 基本推理
        logits = model(images)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        
        print(f"✓ 模型输出形状: {logits.shape}")
        print(f"✓ 预测结果: {predictions.tolist()}")
        print(f"✓ 最高概率: {probabilities.max(dim=-1)[0].tolist()}")
        
        # 获取注意力权重
        logits_with_attn, attention_weights = model(images, return_attention=True)
        print(f"✓ 注意力层数: {len(attention_weights)}")
        print(f"✓ 注意力权重形状: {attention_weights[0].shape}")
    
    # 4. 分析图像分块
    print("\n4. 图像分块分析")
    print("-" * 30)
    
    patch_size = 16
    n_patches = (224 // patch_size) ** 2
    print(f"✓ Patch 大小: {patch_size}x{patch_size}")
    print(f"✓ Patch 数量: {n_patches}")
    print(f"✓ 序列长度: {n_patches + 1} (包含 CLS token)")
    
    # 分析第一张图像的 patch embeddings
    patch_embeddings = model.patch_embed(images[:1])
    cls_token = patch_embeddings[:, 0, :]  # CLS token
    patch_tokens = patch_embeddings[:, 1:, :]  # Patch tokens
    
    print(f"✓ CLS token 形状: {cls_token.shape}")
    print(f"✓ Patch tokens 形状: {patch_tokens.shape}")
    
    # 5. 注意力分析
    print("\n5. 注意力分析")
    print("-" * 30)
    
    # 获取最后一层的注意力
    last_layer_attention = attention_weights[-1]  # [batch, heads, seq_len, seq_len]
    
    # CLS token 对其他 token 的注意力
    cls_attention = last_layer_attention[0, :, 0, 1:]  # [heads, n_patches]
    
    print(f"✓ 最后一层注意力形状: {last_layer_attention.shape}")
    print(f"✓ CLS token 注意力统计:")
    print(f"  - 最大值: {cls_attention.max().item():.4f}")
    print(f"  - 最小值: {cls_attention.min().item():.4f}")
    print(f"  - 平均值: {cls_attention.mean().item():.4f}")
    
    # 找出 CLS token 最关注的 patches
    avg_cls_attention = cls_attention.mean(dim=0)  # 平均所有头
    top_patches = torch.topk(avg_cls_attention, 5)
    
    print(f"✓ CLS token 最关注的 5 个 patches:")
    for i, (score, patch_idx) in enumerate(zip(top_patches.values, top_patches.indices)):
        row = patch_idx // 14  # 14x14 patches
        col = patch_idx % 14
        print(f"  {i+1}. Patch ({row}, {col}): 注意力 = {score:.4f}")
    
    # 6. 模型变体比较
    print("\n6. 模型变体比较")
    print("-" * 30)
    
    model_variants = ['vit_tiny', 'vit_small', 'vit_base']
    
    print(f"{'模型':<12} {'嵌入维度':<8} {'层数':<6} {'头数':<6} {'参数数量':<12}")
    print("-" * 50)
    
    for variant in model_variants:
        try:
            temp_model = create_vit_model(model_name=variant, num_classes=10)
            params, _ = count_parameters(temp_model)
            
            embed_dim = temp_model.embed_dim
            n_layers = temp_model.n_layers
            n_heads = temp_model.transformer_blocks[0].attention.n_heads
            
            print(f"{variant:<12} {embed_dim:<8} {n_layers:<6} {n_heads:<6} {params:<12,}")
            
        except Exception as e:
            print(f"{variant:<12} 创建失败: {e}")
    
    print("\n✅ ViT 快速入门演示完成！")
    return model, images, attention_weights


def visualize_attention_pattern(model, images, attention_weights, save_plots=False):
    """可视化注意力模式"""
    print("\n" + "=" * 60)
    print("注意力模式可视化")
    print("=" * 60)
    
    # 使用第一张图像
    img_idx = 0
    
    # 获取不同层的注意力
    n_layers = len(attention_weights)
    layers_to_show = [0, n_layers//2, n_layers-1]  # 第一层、中间层、最后一层
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, layer_idx in enumerate(layers_to_show):
        # CLS token 的注意力
        cls_attn = attention_weights[layer_idx][img_idx, :, 0, 1:]  # [heads, n_patches]
        cls_attn_avg = cls_attn.mean(dim=0).detach().numpy()  # 平均所有头
        
        # 重塑为2D网格 (14x14)
        grid_size = int(np.sqrt(len(cls_attn_avg)))
        attn_grid = cls_attn_avg.reshape(grid_size, grid_size)
        
        # 绘制注意力热力图
        im1 = axes[0, i].imshow(attn_grid, cmap='hot', interpolation='nearest')
        axes[0, i].set_title(f'第 {layer_idx+1} 层 CLS 注意力')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
        
        # 平均注意力（所有token对所有token）
        avg_attn = attention_weights[layer_idx][img_idx].mean(dim=0)[1:, 1:].detach().numpy()
        
        # 显示部分注意力矩阵
        sample_size = min(49, avg_attn.shape[0])  # 显示7x7的子矩阵
        sample_attn = avg_attn[:sample_size, :sample_size]
        
        im2 = axes[1, i].imshow(sample_attn, cmap='viridis', interpolation='nearest')
        axes[1, i].set_title(f'第 {layer_idx+1} 层 Patch 间注意力')
        axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('vit/attention_patterns.png', dpi=300, bbox_inches='tight')
        print("✓ 注意力模式图已保存到 vit/attention_patterns.png")
    
    try:
        plt.show()
    except:
        print("注意力可视化完成（无法显示图形界面）")
    
    return fig


def demonstrate_different_patch_sizes():
    """演示不同 patch 大小的影响"""
    print("\n" + "=" * 60)
    print("不同 Patch 大小的影响")
    print("=" * 60)
    
    img_size = 224
    patch_sizes = [8, 16, 32]
    test_image = torch.randn(1, 3, img_size, img_size)
    
    print(f"{'Patch大小':<10} {'Patch数量':<10} {'序列长度':<10} {'计算复杂度':<12}")
    print("-" * 50)
    
    for patch_size in patch_sizes:
        n_patches = (img_size // patch_size) ** 2
        seq_len = n_patches + 1  # +1 for CLS token
        
        # 估算注意力计算复杂度 (O(n²))
        attention_ops = seq_len ** 2
        
        # 创建模型测试
        try:
            model = create_vit_model(
                model_name='vit_tiny',
                patch_size=patch_size,
                img_size=img_size,
                num_classes=10
            )
            
            # 测试推理时间
            import time
            model.eval()
            
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):  # 多次测试取平均
                    _ = model(test_image)
            avg_time = (time.time() - start_time) / 10
            
            print(f"{patch_size:<10} {n_patches:<10} {seq_len:<10} {attention_ops:<12,} ({avg_time:.3f}s)")
            
        except Exception as e:
            print(f"{patch_size:<10} 测试失败: {e}")
    
    print("\n观察:")
    print("- 更小的 patch 提供更细粒度的信息，但计算量更大")
    print("- 更大的 patch 计算效率更高，但可能丢失细节")
    print("- patch_size=16 是常用的平衡选择")


def main():
    """主函数：运行完整的快速入门演示"""
    try:
        # 基本演示
        model, images, attention_weights = quick_demo()
        
        # 可视化注意力（如果可能）
        try:
            visualize_attention_pattern(model, images, attention_weights, save_plots=True)
        except Exception as e:
            print(f"注意力可视化跳过: {e}")
        
        # 演示不同 patch 大小
        demonstrate_different_patch_sizes()
        
        print("\n🎉 ViT 快速入门完成！")
        print("\n📚 接下来可以：")
        print("1. 运行 vit_components.py 深入了解各个组件")
        print("2. 运行 patch_embedding.py 学习图像嵌入过程")
        print("3. 运行 vit_model.py 测试完整模型")
        print("4. 运行 vit_trainer.py 进行实际训练")
        print("5. 阅读 README_vit_tutorial.md 获取详细教程")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print("请检查依赖是否正确安装")


if __name__ == "__main__":
    main()
