"""
BERT 快速入门演示
展示 BERT 模型的基本使用和功能
"""

import torch
import torch.nn.functional as F
from bert_model import (
    BERTConfig, BERTModel, BERTForSequenceClassification, 
    BERTForTokenClassification, BERTForQuestionAnswering
)
from bert_pretraining import SimpleTokenizer

def quick_demo():
    """快速演示 BERT 的核心功能"""
    print("🤖 BERT 快速入门演示")
    print("=" * 60)
    
    print("💡 BERT (Bidirectional Encoder Representations from Transformers)")
    print("   是 Google 在 2018 年提出的预训练语言模型")
    print("   通过双向 Transformer 编码器学习深层文本表示")
    print()
    
    # 创建小型配置用于演示
    config = BERTConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=512,
        max_position_embeddings=128
    )
    
    print(f"📋 模型配置:")
    print(f"   词汇表大小: {config.vocab_size:,}")
    print(f"   隐藏层维度: {config.hidden_size}")
    print(f"   编码器层数: {config.num_hidden_layers}")
    print(f"   注意力头数: {config.num_attention_heads}")
    print(f"   前馈网络维度: {config.intermediate_size}")
    
    # 1. 基础 BERT 模型演示
    demonstrate_base_bert(config)
    
    # 2. 文本分类演示
    demonstrate_text_classification(config)
    
    # 3. 词级分类演示
    demonstrate_token_classification(config)
    
    # 4. 问答系统演示
    demonstrate_question_answering(config)
    
    # 5. 注意力可视化
    demonstrate_attention_visualization(config)
    
    print("\n🎉 BERT 快速入门演示完成！")
    print("=" * 60)


def demonstrate_base_bert(config):
    """演示基础 BERT 模型"""
    print("\n🏗️ 1. 基础 BERT 模型演示")
    print("-" * 40)
    
    # 创建模型
    model = BERTModel(config)
    tokenizer = SimpleTokenizer()
    
    # 创建示例输入
    sentence_a = "人工智能改变世界"
    sentence_b = "机器学习很有趣"
    
    print(f"输入句子A: {sentence_a}")
    print(f"输入句子B: {sentence_b}")
    
    # 构造 BERT 输入
    tokens = ['[CLS]'] + sentence_a.split() + ['[SEP]'] + sentence_b.split() + ['[SEP]']
    input_ids = [tokenizer.get(token, 4) for token in tokens]
    token_type_ids = [0] * 6 + [1] * 5  # 句子A为0，句子B为1
    attention_mask = [1] * len(input_ids)
    
    # 填充到统一长度
    max_len = 20
    input_ids.extend([3] * (max_len - len(input_ids)))  # 用PAD填充
    token_type_ids.extend([1] * (max_len - len(token_type_ids)))
    attention_mask.extend([0] * (max_len - len(attention_mask)))
    
    # 转换为张量
    input_ids = torch.tensor([input_ids])
    token_type_ids = torch.tensor([token_type_ids])
    attention_mask = torch.tensor([attention_mask])
    
    print(f"\n输入处理:")
    print(f"  输入tokens: {tokens}")
    print(f"  输入形状: {input_ids.shape}")
    print(f"  token类型: {token_type_ids[0][:len(tokens)]}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        sequence_output, pooled_output = model(input_ids, attention_mask, token_type_ids)
    
    print(f"\n模型输出:")
    print(f"  序列表示形状: {sequence_output.shape}")
    print(f"  池化表示形状: {pooled_output.shape}")
    print(f"  [CLS] token表示: {pooled_output[0][:5].tolist()}")  # 只显示前5个值
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数总量: {total_params:,}")


def demonstrate_text_classification(config):
    """演示文本分类任务"""
    print("\n📝 2. 文本分类演示 (情感分析)")
    print("-" * 40)
    
    # 创建分类模型
    num_labels = 3  # 正面、负面、中性
    model = BERTForSequenceClassification(config, num_labels=num_labels)
    tokenizer = SimpleTokenizer()
    
    # 示例文本
    texts = [
        "这个产品真的很棒",
        "服务态度太差了",
        "还行吧没什么特别的"
    ]
    
    labels = ["正面", "负面", "中性"]
    
    print("示例文本分类:")
    
    for i, text in enumerate(texts):
        # 构造输入
        tokens = ['[CLS]'] + text.split() + ['[SEP]']
        input_ids = [tokenizer.get(token, 4) for token in tokens]
        
        # 填充
        max_len = 15
        input_ids.extend([3] * (max_len - len(input_ids)))
        attention_mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
        
        # 转换为张量
        input_ids = torch.tensor([input_ids])
        attention_mask = torch.tensor([attention_mask])
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            logits = model(input_ids, attention_mask)[0]
            probabilities = F.softmax(logits, dim=-1)
            predicted_label = torch.argmax(logits, dim=-1).item()
        
        print(f"  文本: '{text}'")
        print(f"    预测类别: {labels[predicted_label]}")
        print(f"    置信度分布: {probabilities[0].tolist()}")
        print()


def demonstrate_token_classification(config):
    """演示词级分类任务"""
    print("\n🏷️ 3. 词级分类演示 (命名实体识别)")
    print("-" * 40)
    
    # 创建词级分类模型
    num_labels = 5  # O, B-PER, I-PER, B-ORG, I-ORG
    model = BERTForTokenClassification(config, num_labels=num_labels)
    tokenizer = SimpleTokenizer()
    
    # 示例文本
    text = "张三在北京大学工作"
    tokens = text.split()
    
    print(f"输入文本: {text}")
    print(f"分词结果: {tokens}")
    
    # 构造输入
    bert_tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = [tokenizer.get(token, 4) for token in bert_tokens]
    
    # 填充
    max_len = 15
    input_ids.extend([3] * (max_len - len(input_ids)))
    attention_mask = [1] * len(bert_tokens) + [0] * (max_len - len(bert_tokens))
    
    # 转换为张量
    input_ids = torch.tensor([input_ids])
    attention_mask = torch.tensor([attention_mask])
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)[0]
        predictions = torch.argmax(logits, dim=-1)
    
    # 标签映射
    label_map = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG'}
    
    print(f"\n词级分类结果:")
    for i, token in enumerate(tokens):
        pred_label = predictions[0][i + 1].item()  # +1 跳过[CLS]
        print(f"  {token}: {label_map[pred_label]}")


def demonstrate_question_answering(config):
    """演示问答系统"""
    print("\n❓ 4. 问答系统演示")
    print("-" * 40)
    
    # 创建问答模型
    model = BERTForQuestionAnswering(config)
    tokenizer = SimpleTokenizer()
    
    # 示例问答对
    question = "BERT是什么时候提出的"
    context = "BERT是Google在2018年提出的预训练语言模型它通过双向编码学习文本表示"
    
    print(f"问题: {question}")
    print(f"上下文: {context}")
    
    # 构造输入序列
    question_tokens = question.split()
    context_tokens = context.split()
    
    tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + context_tokens + ['[SEP]']
    input_ids = [tokenizer.get(token, 4) for token in tokens]
    
    # Token type IDs: question为0, context为1
    token_type_ids = [0] * (len(question_tokens) + 2) + [1] * (len(context_tokens) + 1)
    
    # 填充
    max_len = 30
    input_ids.extend([3] * (max_len - len(input_ids)))
    token_type_ids.extend([1] * (max_len - len(token_type_ids)))
    attention_mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
    
    # 转换为张量
    input_ids = torch.tensor([input_ids])
    token_type_ids = torch.tensor([token_type_ids])
    attention_mask = torch.tensor([attention_mask])
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)
        
        start_pos = torch.argmax(start_logits, dim=-1).item()
        end_pos = torch.argmax(end_logits, dim=-1).item()
    
    print(f"\n问答结果:")
    print(f"  预测起始位置: {start_pos}")
    print(f"  预测结束位置: {end_pos}")
    
    # 提取答案
    if start_pos <= end_pos and start_pos < len(tokens) and end_pos < len(tokens):
        answer_tokens = tokens[start_pos:end_pos + 1]
        answer = ' '.join(answer_tokens)
        print(f"  预测答案: {answer}")
    else:
        print(f"  未找到有效答案")


def demonstrate_attention_visualization(config):
    """演示注意力可视化"""
    print("\n👁️ 5. 注意力机制可视化")
    print("-" * 40)
    
    # 创建模型
    model = BERTModel(config)
    tokenizer = SimpleTokenizer()
    
    # 示例句子
    text = "人工智能改变世界"
    tokens = ['[CLS]'] + text.split() + ['[SEP]']
    
    print(f"分析句子: {text}")
    print(f"Token序列: {tokens}")
    
    # 构造输入
    input_ids = [tokenizer.get(token, 4) for token in tokens]
    
    # 填充
    max_len = 10
    input_ids.extend([3] * (max_len - len(input_ids)))
    attention_mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
    
    # 转换为张量
    input_ids = torch.tensor([input_ids])
    attention_mask = torch.tensor([attention_mask])
    
    # 获取注意力权重
    model.eval()
    with torch.no_grad():
        _, _, attention_weights = model(
            input_ids, attention_mask, return_attention=True
        )
    
    print(f"\n注意力分析:")
    print(f"  编码器层数: {len(attention_weights)}")
    print(f"  每层注意力头数: {attention_weights[0].shape[1]}")
    
    # 分析第一层第一个头的注意力
    first_layer_first_head = attention_weights[0][0, 0]  # [seq_len, seq_len]
    
    print(f"\n第1层第1头注意力矩阵:")
    print("  " + " ".join([f"{token:>6}" for token in tokens]))
    
    for i, token in enumerate(tokens):
        attention_scores = first_layer_first_head[i, :len(tokens)]
        scores_str = " ".join([f"{score:.3f}" for score in attention_scores])
        print(f"{token:>6}: {scores_str}")
    
    # 找出最高注意力的token对
    max_attention = first_layer_first_head[:len(tokens), :len(tokens)].max()
    max_pos = torch.where(first_layer_first_head[:len(tokens), :len(tokens)] == max_attention)
    
    if len(max_pos[0]) > 0:
        from_token = tokens[max_pos[0][0].item()]
        to_token = tokens[max_pos[1][0].item()]
        print(f"\n最高注意力: {from_token} -> {to_token} ({max_attention:.4f})")


def compare_bert_variants():
    """比较不同 BERT 模型变体"""
    print("\n🔬 6. BERT 模型变体比较")
    print("-" * 40)
    
    configs = {
        "BERT-Tiny": BERTConfig(
            vocab_size=1000, hidden_size=128, num_hidden_layers=2,
            num_attention_heads=2, intermediate_size=256
        ),
        "BERT-Small": BERTConfig(
            vocab_size=1000, hidden_size=256, num_hidden_layers=4,
            num_attention_heads=4, intermediate_size=512
        ),
        "BERT-Medium": BERTConfig(
            vocab_size=1000, hidden_size=512, num_hidden_layers=8,
            num_attention_heads=8, intermediate_size=1024
        )
    }
    
    print("模型规模对比:")
    print(f"{'模型':>12} {'隐层维度':>8} {'层数':>4} {'头数':>4} {'参数量':>10}")
    print("-" * 50)
    
    for name, config in configs.items():
        model = BERTModel(config)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:>12} {config.hidden_size:>8} {config.num_hidden_layers:>4} "
              f"{config.num_attention_heads:>4} {params:>10,}")
    
    return configs


def demonstrate_key_differences():
    """演示 BERT 与其他模型的关键区别"""
    print("\n🆚 7. BERT 关键特性对比")
    print("-" * 40)
    
    print("🔄 双向编码 vs 单向编码:")
    print("  传统模型: 从左到右 (或从右到左) 单向处理")
    print("  BERT: 同时考虑左右上下文，获得更丰富的表示")
    print()
    
    print("🎯 预训练任务:")
    print("  MLM (掩码语言模型): 预测被掩盖的词")
    print("  NSP (下一句预测): 判断句子间的逻辑关系")
    print()
    
    print("🔧 迁移学习范式:")
    print("  预训练: 在大规模无标注语料上学习通用表示")
    print("  微调: 在特定任务数据上调整参数")
    print()
    
    print("🏗️ 架构特点:")
    print("  仅编码器: 专注于文本理解任务")
    print("  多层 Transformer: 深层特征提取")
    print("  特殊 Token: [CLS], [SEP], [MASK] 等")


if __name__ == "__main__":
    print("🚀 开始 BERT 快速入门演示")
    
    try:
        # 主要演示
        quick_demo()
        
        # 额外对比
        compare_bert_variants()
        demonstrate_key_differences()
        
        print("\n✨ 总结:")
        print("1. BERT 通过双向编码学习丰富的文本表示")
        print("2. 预训练 + 微调范式适配多种下游任务")
        print("3. 注意力机制揭示词与词之间的关系")
        print("4. 不同规模的模型适合不同的应用场景")
        
        print("\n🎓 学习建议:")
        print("- 理解 Transformer 编码器结构")
        print("- 掌握 MLM 和 NSP 预训练任务")
        print("- 实践不同下游任务的微调")
        print("- 分析注意力权重理解模型行为")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {str(e)}")
        print("请检查模型组件是否正确实现")
    
    print("\n🎉 BERT 快速入门演示结束!")
