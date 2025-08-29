# Vision Transformer (ViT) 完整教程

## 目录
1. [ViT 基础概念](#vit-基础概念)
2. [图像分块机制](#图像分块机制)
3. [位置编码](#位置编码)
4. [Transformer 编码器](#transformer-编码器)
5. [分类头](#分类头)
6. [完整实现](#完整实现)
7. [训练示例](#训练示例)

## ViT 基础概念

Vision Transformer (ViT) 是 Google 在 2020 年提出的模型，将 Transformer 架构成功应用到图像分类任务中。

### 核心思想
1. **图像分块**：将图像切分成固定大小的 patch
2. **线性投影**：将每个 patch 映射到嵌入向量
3. **位置编码**：为每个 patch 添加位置信息
4. **Transformer 编码**：使用标准 Transformer 编码器处理序列
5. **分类预测**：通过分类头输出最终预测

### ViT vs CNN
| 特性 | CNN | ViT |
|------|-----|-----|
| 归纳偏置 | 局部性、平移不变性 | 较少归纳偏置 |
| 感受野 | 逐层增大 | 全局 |
| 数据需求 | 中等 | 大量 |
| 可解释性 | 特征图 | 注意力图 |

## 图像分块机制

ViT 的第一步是将图像切分成不重叠的 patch：

```
输入图像: (H, W, C)
Patch 大小: (P, P)
Patch 数量: N = HW/P²
每个 patch: (P, P, C)
展平后: (P²·C,)
```

### 数学表示
```
x ∈ R^(H×W×C) → x_p ∈ R^(N×(P²·C))
```

其中 N = HW/P² 是 patch 的数量。

## 位置编码

由于 Transformer 无法感知位置信息，需要为每个 patch 添加位置编码：

1. **1D 位置编码**：简单的可学习嵌入
2. **2D 位置编码**：考虑 patch 的行列位置
3. **相对位置编码**：编码 patch 之间的相对关系

## Transformer 编码器

ViT 使用标准的 Transformer 编码器：
- 多头自注意力机制
- 层归一化
- 残差连接
- MLP 前馈网络

## 分类头

ViT 使用特殊的 [CLS] token 进行分类：
1. 在 patch 序列前添加可学习的 [CLS] token
2. 经过 Transformer 编码后，提取 [CLS] token 的表示
3. 通过 MLP 头输出分类结果

## 文件结构

```
vit/
├── README_vit_tutorial.md          # 本教程文档
├── vit_components.py               # ViT 基础组件
├── patch_embedding.py              # 图像分块和嵌入
├── vit_model.py                   # 完整 ViT 模型
├── vit_trainer.py                 # 训练和评估
├── attention_visualization.py      # 注意力可视化
├── feature_visualization.py       # 特征可视化
├── quick_start.py                 # 快速入门
└── run_examples.py                # 示例运行器
```

## 学习路径

### 🚀 快速入门（30分钟）
1. 运行 `quick_start.py` 了解 ViT 基本功能
2. 阅读本教程理解核心概念
3. 查看 `vit_components.py` 理解组件实现

### 📚 深入学习（2-3小时）
1. 学习图像分块机制 (`patch_embedding.py`)
2. 理解 ViT 完整架构 (`vit_model.py`)
3. 体验训练过程 (`vit_trainer.py`)
4. 分析注意力模式 (`attention_visualization.py`)

### 🔬 研究路径（一天）
1. 完成深入学习的所有内容
2. 可视化学习特征和注意力
3. 尝试不同的 patch 大小和模型配置
4. 实现自己的改进版本

## 依赖安装

```bash
pip install torch torchvision numpy matplotlib seaborn pillow
```

## 使用方法

```bash
# 快速入门
python vit/quick_start.py

# 完整训练
python vit/vit_trainer.py

# 注意力可视化
python vit/attention_visualization.py

# 交互式学习
python vit/run_examples.py
```

## ViT 变体

1. **ViT-Base**: 12层，768维，12头
2. **ViT-Large**: 24层，1024维，16头  
3. **ViT-Huge**: 32层，1280维，16头
4. **DeiT**: 数据高效的 ViT
5. **Swin Transformer**: 层次化 ViT

这个教程将帮助你完全理解 ViT 的工作原理和实现细节！
