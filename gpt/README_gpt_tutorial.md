# 🤖 GPT (Generative Pre-trained Transformer) 完整教程

> 从 Transformer 到 GPT：理解生成式语言模型的核心原理与实现

## 🎯 学习目标

通过本教程，您将：
- 🧠 **深入理解** GPT 架构与 Transformer 的区别
- 💻 **完整实现** GPT-1/2 级别的模型
- 🎯 **掌握训练技巧** 语言建模、文本生成
- 🔍 **学会分析** 生成质量、困惑度评估
- 🚀 **体验完整流程** 从训练到推理的全过程

## 📚 理论基础

### 🔄 GPT vs Transformer

| 特性 | Transformer | GPT |
|------|-------------|-----|
| **架构** | 编码器-解码器 | 仅解码器 |
| **注意力** | 双向注意力 | 因果(单向)注意力 |
| **任务** | 序列到序列 | 语言建模 |
| **训练目标** | 翻译等监督任务 | 下一词预测 |

### 🧠 核心创新点

#### 1. 因果注意力掩码
```python
# 确保模型只能看到当前位置之前的信息
mask = torch.tril(torch.ones(seq_len, seq_len))
# 上三角部分设为 -inf，softmax 后变为 0
attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
```

#### 2. 自回归生成
```python
# 逐个生成下一个词
for _ in range(max_length):
    logits = model(input_ids)
    next_token = sample_from_logits(logits[:, -1, :])
    input_ids = torch.cat([input_ids, next_token], dim=1)
```

#### 3. 无监督预训练
```python
# 损失函数：预测下一个词
loss = CrossEntropyLoss()(
    logits.view(-1, vocab_size), 
    targets.view(-1)
)
```

## 🏗️ 架构设计

### 📐 整体结构
```
输入: "The cat sat on"
     ↓
词嵌入 + 位置编码
     ↓
GPT Block 1 (因果自注意力 + FFN)
     ↓
GPT Block 2 (因果自注意力 + FFN)
     ↓
...
     ↓
Layer Norm
     ↓
语言建模头 (Linear)
     ↓
输出概率: ["the", "a", "mat", ...]
```

### 🧩 核心组件

#### 1. 因果多头注意力
- Q、K、V 投影
- 因果掩码防止"未来泄露"
- 多头并行计算

#### 2. 位置前馈网络
- 两层线性变换
- GELU 激活函数
- Dropout 正则化

#### 3. 残差连接与层归一化
- Pre-LN：先归一化再计算
- 残差连接稳定训练

## 🎯 训练策略

### 📊 数据准备
1. **文本清洗** - 去除特殊字符、统一格式
2. **分词处理** - BPE/WordPiece 子词分割
3. **序列构建** - 滑动窗口创建训练样本

### 🏋️ 训练技巧
1. **梯度累积** - 模拟大批次训练
2. **学习率调度** - Warmup + Cosine Annealing
3. **梯度裁剪** - 防止梯度爆炸
4. **权重衰减** - L2 正则化

### 📈 评估指标
1. **困惑度 (Perplexity)** - 模型预测不确定性
2. **BLEU Score** - 生成质量评估
3. **人工评估** - 流畅性、连贯性

## 🚀 生成策略

### 🎲 采样方法

#### 1. 贪心解码
```python
next_token = torch.argmax(logits, dim=-1)
```

#### 2. 随机采样
```python
probs = F.softmax(logits / temperature, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

#### 3. Top-K 采样
```python
top_k_logits, top_k_indices = torch.topk(logits, k=50)
probs = F.softmax(top_k_logits / temperature, dim=-1)
next_token = top_k_indices[torch.multinomial(probs, 1)]
```

#### 4. Top-P (Nucleus) 采样
```python
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
nucleus_mask = cumulative_probs <= p
```

## 🔬 实验设计

### 🧪 消融实验
1. **模型大小** - 参数量对性能的影响
2. **层数深度** - 深度与表达能力的关系
3. **注意力头数** - 多头注意力的作用
4. **上下文长度** - 序列长度的影响

### 📊 对比实验
1. **vs LSTM** - 传统循环网络对比
2. **vs BERT** - 双向 vs 单向编码
3. **vs Transformer** - 编码器-解码器对比

## 💡 应用场景

### ✍️ 文本生成
- **创意写作** - 小说、诗歌创作
- **对话生成** - 聊天机器人
- **内容创作** - 文章、广告文案

### 🔧 下游任务
- **文本分类** - 情感分析、主题分类
- **问答系统** - 阅读理解、知识问答
- **摘要生成** - 自动摘要、关键词提取

## 🛠️ 实践项目

### 项目1: 小说生成器
- 训练数据：经典小说文本
- 目标：生成连贯的故事情节
- 评估：人工评分 + 困惑度

### 项目2: 代码生成助手  
- 训练数据：GitHub 代码库
- 目标：根据注释生成代码
- 评估：代码正确性 + 风格一致性

### 项目3: 对话聊天机器人
- 训练数据：对话语料库
- 目标：生成自然的对话回复
- 评估：对话质量 + 上下文一致性

## 📈 优化技巧

### ⚡ 训练加速
1. **混合精度训练** - FP16 + FP32
2. **梯度检查点** - 内存与计算的权衡
3. **数据并行** - 多GPU 训练
4. **模型并行** - 大模型分片

### 🎯 生成优化
1. **Beam Search** - 多路径搜索
2. **长度惩罚** - 避免过短/过长生成
3. **重复惩罚** - 减少重复内容
4. **内容过滤** - 避免有害内容

## 🔮 进阶方向

### 🌟 模型改进
1. **Sparse Attention** - 减少注意力计算
2. **Retrieval Augmented** - 检索增强生成
3. **Multimodal GPT** - 多模态输入
4. **Instruction Tuning** - 指令微调

### 🚀 应用扩展
1. **Few-shot Learning** - 少样本学习
2. **Chain of Thought** - 思维链推理
3. **Tool Using** - 工具调用能力
4. **Agent Behavior** - 智能体行为

---

## 📝 学习检查点

完成本教程后，您应该能够：

### 理论理解 ✅
- [ ] 解释 GPT 与 Transformer 的核心区别
- [ ] 描述因果注意力的工作原理
- [ ] 理解自回归生成的过程
- [ ] 掌握各种采样方法的优缺点

### 实践能力 ✅
- [ ] 从零实现 GPT 模型
- [ ] 训练语言模型并评估性能
- [ ] 实现多种文本生成策略
- [ ] 调试和优化模型性能

### 应用技能 ✅
- [ ] 构建实际的文本生成应用
- [ ] 评估生成质量并改进
- [ ] 部署模型进行在线推理
- [ ] 设计用户交互界面

---

## 🎉 下一步学习

完成 GPT 基础后，建议继续学习：
1. **🔥 GPT-3/4 级别的大模型训练**
2. **🎯 指令微调 (Instruction Tuning)**
3. **🤖 人类反馈强化学习 (RLHF)**
4. **🌐 多模态 GPT (GPT-4V)**

让我们开始实践吧！🚀
