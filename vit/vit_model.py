"""
完整的 Vision Transformer (ViT) 模型实现
包含从图像输入到分类输出的完整流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from vit_components import TransformerBlock
from patch_embedding import ViTEmbedding

class VisionTransformer(nn.Module):
    """
    Vision Transformer 完整模型
    
    ViT 的核心架构：
    1. 图像分块和嵌入 (Patch Embedding)
    2. 多层 Transformer 编码器 
    3. 分类头 (Classification Head)
    """
    
    def __init__(self, 
                 img_size=224,           # 输入图像尺寸
                 patch_size=16,          # 图像块大小
                 in_channels=3,          # 输入通道数
                 num_classes=1000,       # 分类类别数
                 embed_dim=768,          # 嵌入维度
                 n_layers=12,            # Transformer 层数
                 n_heads=12,             # 注意力头数
                 mlp_ratio=4.0,          # MLP 隐藏层倍数
                 dropout=0.1,            # Dropout 概率
                 attention_dropout=0.1,  # 注意力 Dropout
                 drop_path_rate=0.1):    # Drop Path 概率
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        
        # 计算MLP隐藏层维度
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        
        # 1. 图像嵌入层
        self.patch_embed = ViTEmbedding(
            img_size=img_size,
            patch_size=patch_size, 
            in_channels=in_channels,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # 2. Transformer 编码器层
        # 使用 drop_path_rate 创建递增的 drop path 概率
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=embed_dim,
                n_heads=n_heads,
                d_ff=mlp_hidden_dim,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # 3. 层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 4. 分类头
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 5. 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            # 使用截断正态分布初始化线性层
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            # 使用fan_out模式的正态分布初始化卷积层
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward_features(self, x):
        """
        特征提取前向传播
        
        Args:
            x: 输入图像 [batch_size, channels, height, width]
            
        Returns:
            cls_token: CLS token 特征 [batch_size, embed_dim]
            patch_features: Patch 特征 [batch_size, n_patches, embed_dim]
            attention_weights: 各层注意力权重
        """
        # 1. 图像嵌入
        x = self.patch_embed(x)  # [B, n_patches + 1, embed_dim]
        
        # 2. 通过Transformer层
        attention_weights = []
        for block in self.transformer_blocks:
            x, attn = block(x)
            attention_weights.append(attn)
        
        # 3. 层归一化
        x = self.norm(x)
        
        # 4. 分离CLS token和patch features
        cls_token = x[:, 0]  # [B, embed_dim]
        patch_features = x[:, 1:]  # [B, n_patches, embed_dim]
        
        return cls_token, patch_features, attention_weights
    
    def forward(self, x, return_attention=False):
        """
        完整前向传播
        
        Args:
            x: 输入图像 [batch_size, channels, height, width]
            return_attention: 是否返回注意力权重
            
        Returns:
            logits: 分类预测 [batch_size, num_classes]
            attention_weights: 注意力权重 (如果 return_attention=True)
        """
        # 特征提取
        cls_token, patch_features, attention_weights = self.forward_features(x)
        
        # 分类预测
        logits = self.head(cls_token)
        
        if return_attention:
            return logits, attention_weights
        else:
            return logits
    
    def get_attention_maps(self, x, layer_idx=-1, head_idx=None):
        """
        获取指定层的注意力图
        
        Args:
            x: 输入图像
            layer_idx: 层索引 (-1 表示最后一层)
            head_idx: 头索引 (None 表示平均所有头)
            
        Returns:
            attention_map: 注意力图
        """
        _, _, attention_weights = self.forward_features(x)
        
        # 获取指定层的注意力权重
        attn = attention_weights[layer_idx]  # [B, n_heads, seq_len, seq_len]
        
        if head_idx is not None:
            # 返回指定头的注意力
            attn = attn[:, head_idx]  # [B, seq_len, seq_len]
        else:
            # 平均所有头
            attn = attn.mean(dim=1)  # [B, seq_len, seq_len]
        
        return attn
    
    def interpolate_pos_encoding(self, x, h, w):
        """
        插值位置编码以适应不同尺寸的图像
        
        Args:
            x: patch embeddings
            h, w: 新的高度和宽度 (以patch为单位)
        """
        n_patches = x.shape[1] - 1
        if n_patches == h * w:
            return x
        
        # 获取位置编码
        pos_embed = self.patch_embed.position_encoding.position_embeddings
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        
        # 重塑为2D
        dim = x.shape[-1]
        h0 = w0 = int(math.sqrt(n_patches))
        patch_pos_embed = patch_pos_embed.reshape(1, h0, w0, dim).permute(0, 3, 1, 2)
        
        # 插值到新尺寸
        patch_pos_embed = F.interpolate(
            patch_pos_embed, size=(h, w), mode='bicubic', align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
        
        # 重新组合
        return torch.cat([class_pos_embed.unsqueeze(1), patch_pos_embed], dim=1)


class ViT_Tiny(VisionTransformer):
    """ViT-Tiny 配置"""
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=192,
            n_layers=12,
            n_heads=3,
            mlp_ratio=4.0,
            **kwargs
        )


class ViT_Small(VisionTransformer):
    """ViT-Small 配置"""
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=384,
            n_layers=12,
            n_heads=6,
            mlp_ratio=4.0,
            **kwargs
        )


class ViT_Base(VisionTransformer):
    """ViT-Base 配置 (默认)"""
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=768,
            n_layers=12,
            n_heads=12,
            mlp_ratio=4.0,
            **kwargs
        )


class ViT_Large(VisionTransformer):
    """ViT-Large 配置"""
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=1024,
            n_layers=24,
            n_heads=16,
            mlp_ratio=4.0,
            **kwargs
        )


class ViT_Huge(VisionTransformer):
    """ViT-Huge 配置"""
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=1280,
            n_layers=32,
            n_heads=16,
            mlp_ratio=4.0,
            **kwargs
        )


def create_vit_model(model_name='vit_base', **kwargs):
    """
    创建指定配置的ViT模型
    
    Args:
        model_name: 模型名称 ('vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'vit_huge')
        **kwargs: 其他参数
        
    Returns:
        model: ViT模型
    """
    model_configs = {
        'vit_tiny': ViT_Tiny,
        'vit_small': ViT_Small,
        'vit_base': ViT_Base,
        'vit_large': ViT_Large,
        'vit_huge': ViT_Huge
    }
    
    if model_name not in model_configs:
        raise ValueError(f"未知模型: {model_name}. 可选: {list(model_configs.keys())}")
    
    return model_configs[model_name](**kwargs)


def count_parameters(model):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def test_vit_models():
    """测试不同配置的ViT模型"""
    print("=" * 60)
    print("测试 ViT 模型")
    print("=" * 60)
    
    # 测试参数
    batch_size = 2
    img_size = 224
    in_channels = 3
    num_classes = 1000
    
    # 创建测试输入
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    print(f"输入形状: {x.shape}")
    
    # 测试不同配置的模型
    model_names = ['vit_tiny', 'vit_small', 'vit_base']
    
    for model_name in model_names:
        print(f"\n{'-' * 40}")
        print(f"测试 {model_name.upper()}")
        print(f"{'-' * 40}")
        
        # 创建模型
        model = create_vit_model(
            model_name=model_name,
            img_size=img_size,
            num_classes=num_classes
        )
        
        # 计算参数数量
        total_params, trainable_params = count_parameters(model)
        
        print(f"模型配置:")
        print(f"  嵌入维度: {model.embed_dim}")
        print(f"  层数: {model.n_layers}")
        print(f"  注意力头数: {model.transformer_blocks[0].attention.n_heads}")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        # 前向传播测试
        model.eval()
        with torch.no_grad():
            # 基本前向传播
            logits = model(x)
            print(f"  输出形状: {logits.shape}")
            print(f"  输出统计: 均值={logits.mean():.4f}, 标准差={logits.std():.4f}")
            
            # 带注意力权重的前向传播
            logits_with_attn, attention_weights = model(x, return_attention=True)
            print(f"  注意力层数: {len(attention_weights)}")
            print(f"  注意力权重形状: {attention_weights[0].shape}")
            
            # 测试注意力图
            attn_map = model.get_attention_maps(x, layer_idx=-1)
            print(f"  注意力图形状: {attn_map.shape}")


def demonstrate_model_components():
    """演示模型各组件的功能"""
    print("\n" + "=" * 60)
    print("演示 ViT 模型组件")
    print("=" * 60)
    
    # 创建测试模型
    model = ViT_Base(img_size=224, num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    
    print("1. 完整前向传播流程")
    print("-" * 30)
    
    model.eval()
    with torch.no_grad():
        # 步骤1: 图像嵌入
        embeddings = model.patch_embed(x)
        print(f"图像嵌入后形状: {embeddings.shape}")
        
        # 步骤2: Transformer编码
        cls_token, patch_features, attention_weights = model.forward_features(x)
        print(f"CLS token形状: {cls_token.shape}")
        print(f"Patch特征形状: {patch_features.shape}")
        print(f"注意力层数: {len(attention_weights)}")
        
        # 步骤3: 分类预测
        logits = model.head(cls_token)
        print(f"分类输出形状: {logits.shape}")
        
        # 步骤4: 概率预测
        probs = F.softmax(logits, dim=-1)
        predicted_class = probs.argmax(dim=-1)
        print(f"预测类别: {predicted_class.item()}")
        print(f"最高概率: {probs.max().item():.4f}")


if __name__ == "__main__":
    # 测试不同配置的ViT模型
    test_vit_models()
    
    # 演示模型组件
    demonstrate_model_components()
    
    print("\n" + "=" * 60)
    print("ViT 模型测试完成！")
    print("=" * 60)
