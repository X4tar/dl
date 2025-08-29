"""
BERT 预训练数据处理和训练
包含 MLM 和 NSP 任务的数据准备和模型训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from bert_model import BERTForPreTraining, BERTConfig

class BERTDataset(Dataset):
    """
    BERT 预训练数据集
    处理 MLM 和 NSP 任务的数据
    """
    
    def __init__(self, texts, tokenizer, max_length=128, mlm_probability=0.15):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            tokenizer: 分词器
            max_length: 最大序列长度
            mlm_probability: MLM 掩码概率
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        
        # 特殊 token IDs
        self.cls_token_id = tokenizer.get('[CLS]', 0)
        self.sep_token_id = tokenizer.get('[SEP]', 1)
        self.mask_token_id = tokenizer.get('[MASK]', 2)
        self.pad_token_id = tokenizer.get('[PAD]', 3)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """获取一个训练样本"""
        # 创建句子对和 NSP 标签
        sentence_a, sentence_b, is_next = self.create_sentence_pair(idx)
        
        # 构造输入序列
        input_ids, token_type_ids, attention_mask = self.create_input_sequence(
            sentence_a, sentence_b
        )
        
        # 创建 MLM 标签
        input_ids, mlm_labels = self.create_mlm_labels(input_ids)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'mlm_labels': torch.tensor(mlm_labels, dtype=torch.long),
            'nsp_labels': torch.tensor(is_next, dtype=torch.long)
        }
    
    def create_sentence_pair(self, idx):
        """创建句子对用于 NSP 任务"""
        current_text = self.texts[idx]
        sentences = current_text.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        
        if len(sentences) < 2:
            # 如果句子不够，使用单句
            sentence_a = sentences[0] if sentences else "这是一个示例句子"
            sentence_b = "这是另一个示例句子"
            is_next = 0  # 不是下一句
        else:
            # 50% 概率选择连续句子，50% 概率选择随机句子
            if random.random() < 0.5:
                # 选择连续句子
                start_idx = random.randint(0, len(sentences) - 2)
                sentence_a = sentences[start_idx]
                sentence_b = sentences[start_idx + 1]
                is_next = 1  # 是下一句
            else:
                # 选择随机句子
                sentence_a = random.choice(sentences)
                # 从其他文本中选择句子
                other_idx = random.randint(0, len(self.texts) - 1)
                while other_idx == idx:
                    other_idx = random.randint(0, len(self.texts) - 1)
                other_sentences = self.texts[other_idx].split('.')
                other_sentences = [s.strip() for s in other_sentences if len(s.strip()) > 0]
                sentence_b = random.choice(other_sentences) if other_sentences else "随机句子"
                is_next = 0  # 不是下一句
        
        return sentence_a, sentence_b, is_next
    
    def create_input_sequence(self, sentence_a, sentence_b):
        """构造 BERT 输入序列"""
        # 简单的分词（实际应用中需要更复杂的分词器）
        tokens_a = sentence_a.split()[:50]  # 限制长度
        tokens_b = sentence_b.split()[:50]
        
        # 构造序列: [CLS] tokens_a [SEP] tokens_b [SEP]
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        
        # 截断或填充到指定长度
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens.extend(['[PAD]'] * (self.max_length - len(tokens)))
        
        # 转换为 ID
        input_ids = [self.tokenizer.get(token, 4) for token in tokens]  # 4 为 UNK
        
        # 创建 token type IDs
        token_type_ids = []
        sep_count = 0
        for token in tokens:
            if token == '[SEP]':
                sep_count += 1
            token_type_ids.append(0 if sep_count < 1 else 1)
        
        # 创建 attention mask
        attention_mask = [1 if token != '[PAD]' else 0 for token in tokens]
        
        return input_ids, token_type_ids, attention_mask
    
    def create_mlm_labels(self, input_ids):
        """创建 MLM 标签"""
        input_ids = input_ids.copy()
        mlm_labels = [-100] * len(input_ids)  # -100 表示不计算损失
        
        # 对 15% 的 token 进行掩码
        for i, token_id in enumerate(input_ids):
            # 跳过特殊 token
            if token_id in [self.cls_token_id, self.sep_token_id, self.pad_token_id]:
                continue
            
            if random.random() < self.mlm_probability:
                mlm_labels[i] = token_id  # 保存原始 token 用于计算损失
                
                prob = random.random()
                if prob < 0.8:
                    # 80% 替换为 [MASK]
                    input_ids[i] = self.mask_token_id
                elif prob < 0.9:
                    # 10% 替换为随机 token
                    input_ids[i] = random.randint(5, 999)  # 假设词汇表大小为 1000
                # 10% 保持不变
        
        return input_ids, mlm_labels


class SimpleTokenizer:
    """简单的分词器（用于演示）"""
    
    def __init__(self):
        self.token_to_id = {
            '[CLS]': 0,
            '[SEP]': 1,
            '[MASK]': 2,
            '[PAD]': 3,
            '[UNK]': 4
        }
        self.vocab_size = 1000
        
        # 添加一些常见词汇
        common_words = [
            '的', '是', '在', '了', '不', '和', '有', '人', '这', '中', '大', '为', '上', '个', '国',
            '我', '以', '要', '他', '时', '来', '用', '们', '生', '到', '作', '地', '于', '出', '就',
            'the', 'a', 'to', 'and', 'of', 'is', 'in', 'it', 'you', 'that', 'he', 'was', 'for',
            'on', 'are', 'as', 'with', 'his', 'they', 'i', 'at', 'be', 'this', 'have', 'from'
        ]
        
        for i, word in enumerate(common_words):
            self.token_to_id[word] = i + 5
    
    def get(self, token, default=None):
        return self.token_to_id.get(token, default)


class BERTTrainer:
    """BERT 预训练器"""
    
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        
        # 创建模型
        self.model = BERTForPreTraining(config).to(device)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=1000
        )
        
    def train(self, dataloader, epochs=1):
        """训练模型"""
        self.model.train()
        total_loss = 0
        step = 0
        
        print(f"开始 BERT 预训练，共 {epochs} 个 epoch")
        print("=" * 60)
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # 移动数据到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch['token_type_ids'],
                    masked_lm_labels=batch['mlm_labels'],
                    next_sentence_label=batch['nsp_labels']
                )
                
                loss = outputs[0]
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()
                epoch_loss += loss.item()
                step += 1
                
                # 打印进度
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    lr = self.optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch+1}/{epochs}, Step {batch_idx+1}, "
                          f"Loss: {avg_loss:.4f}, LR: {lr:.6f}")
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} 完成，平均损失: {avg_epoch_loss:.4f}")
            print("-" * 40)
        
        avg_total_loss = total_loss / step
        print(f"预训练完成！平均损失: {avg_total_loss:.4f}")
        
        return avg_total_loss


def create_sample_data():
    """创建示例训练数据"""
    texts = [
        "人工智能是计算机科学的一个分支。它试图理解智能的实质。人工智能的研究历史有着一条从以推理为重点到以知识为重点。",
        "机器学习是人工智能的一个重要分支。它使计算机能够不经过明确编程就能学习。机器学习算法基于样本数据进行训练。",
        "深度学习是机器学习的一个子集。它基于人工神经网络的概念。深度学习在图像识别和自然语言处理方面取得了重大突破。",
        "自然语言处理是人工智能的一个重要领域。它研究如何让计算机理解和生成人类语言。自然语言处理在搜索引擎和翻译软件中广泛应用。",
        "计算机视觉是人工智能的另一个重要分支。它使计算机能够理解和解释视觉信息。计算机视觉在自动驾驶和医学影像分析中有重要应用。",
        "强化学习是机器学习的一种方法。它通过试错学习来获得最佳策略。强化学习在游戏和机器人控制中表现出色。",
        "神经网络是深度学习的基础。它模仿人脑神经元的工作原理。神经网络通过调整权重来学习数据中的模式。",
        "数据科学结合了统计学和计算机科学。它从大量数据中提取有价值的信息。数据科学在商业决策和科学研究中发挥重要作用。"
    ]
    return texts


def demonstrate_bert_pretraining():
    """演示 BERT 预训练过程"""
    print("=" * 60)
    print("BERT 预训练演示")
    print("=" * 60)
    
    # 创建配置（小模型用于演示）
    config = BERTConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=512,
        max_position_embeddings=128
    )
    
    print(f"模型配置:")
    print(f"  词汇表大小: {config.vocab_size}")
    print(f"  隐藏层大小: {config.hidden_size}")
    print(f"  层数: {config.num_hidden_layers}")
    print(f"  注意力头数: {config.num_attention_heads}")
    
    # 创建数据
    texts = create_sample_data()
    tokenizer = SimpleTokenizer()
    
    print(f"\n训练数据:")
    print(f"  文本数量: {len(texts)}")
    print(f"  示例文本: {texts[0][:50]}...")
    
    # 创建数据集和数据加载器
    dataset = BERTDataset(texts, tokenizer, max_length=64)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"\n数据集信息:")
    print(f"  样本数量: {len(dataset)}")
    print(f"  批次大小: 4")
    print(f"  序列长度: 64")
    
    # 创建训练器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = BERTTrainer(config, device)
    
    print(f"\n训练设置:")
    print(f"  设备: {device}")
    print(f"  模型参数: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    # 检查一个批次的数据
    print(f"\n数据样本检查:")
    sample_batch = next(iter(dataloader))
    for key, value in sample_batch.items():
        print(f"  {key}: {value.shape}")
    
    # 训练模型
    print(f"\n开始训练...")
    trainer.train(dataloader, epochs=2)
    
    return trainer


def analyze_bert_attention():
    """分析 BERT 注意力模式"""
    print("\n" + "=" * 60)
    print("BERT 注意力分析")
    print("=" * 60)
    
    # 创建小型配置
    config = BERTConfig(
        vocab_size=100,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=64
    )
    
    # 创建模型
    from bert_model import BERTModel
    model = BERTModel(config)
    model.eval()
    
    # 创建示例输入
    batch_size = 1
    seq_len = 16
    input_ids = torch.randint(5, 50, (batch_size, seq_len))  # 避免特殊token
    attention_mask = torch.ones(batch_size, seq_len)
    token_type_ids = torch.cat([
        torch.zeros(batch_size, 8, dtype=torch.long),
        torch.ones(batch_size, 8, dtype=torch.long)
    ], dim=1)
    
    print(f"输入信息:")
    print(f"  输入形状: {input_ids.shape}")
    print(f"  注意力掩码: {attention_mask.shape}")
    print(f"  Token类型: {token_type_ids.shape}")
    
    # 获取注意力权重
    with torch.no_grad():
        sequence_output, pooled_output, attention_weights = model(
            input_ids, attention_mask, token_type_ids, return_attention=True
        )
    
    print(f"\n输出信息:")
    print(f"  序列输出: {sequence_output.shape}")
    print(f"  池化输出: {pooled_output.shape}")
    print(f"  注意力权重层数: {len(attention_weights)}")
    print(f"  每层注意力形状: {attention_weights[0].shape}")
    
    # 分析第一层的注意力模式
    first_layer_attention = attention_weights[0][0]  # [num_heads, seq_len, seq_len]
    
    print(f"\n第一层注意力分析:")
    print(f"  注意力头数: {first_layer_attention.shape[0]}")
    
    # 分析每个头的注意力分布
    for head_idx in range(min(2, first_layer_attention.shape[0])):
        attention_matrix = first_layer_attention[head_idx]
        
        print(f"\n  头 {head_idx + 1} 注意力分布:")
        print(f"  最大注意力值: {attention_matrix.max().item():.4f}")
        print(f"  注意力熵: {(-attention_matrix * torch.log(attention_matrix + 1e-8)).sum(dim=-1).mean().item():.4f}")
        
        # 找出最被关注的位置
        max_attention_pos = attention_matrix.sum(dim=0).argmax().item()
        print(f"  最被关注位置: {max_attention_pos}")
    
    return attention_weights


def demonstrate_mlm_task():
    """演示 MLM 任务"""
    print("\n" + "=" * 60)
    print("掩码语言模型 (MLM) 任务演示")
    print("=" * 60)
    
    # 创建示例数据
    tokenizer = SimpleTokenizer()
    
    # 原始句子
    original_sentence = "人工智能 是 计算机 科学 的 一个 分支"
    print(f"原始句子: {original_sentence}")
    
    # 简单分词
    tokens = original_sentence.split()
    print(f"分词结果: {tokens}")
    
    # 转换为ID
    token_ids = [tokenizer.get(token, 4) for token in tokens]
    print(f"Token IDs: {token_ids}")
    
    # 创建掩码版本
    masked_tokens = tokens.copy()
    masked_ids = token_ids.copy()
    mlm_labels = [-100] * len(token_ids)
    
    # 随机掩盖一些词
    mask_positions = [1, 4]  # 掩盖 "是" 和 "的"
    for pos in mask_positions:
        if pos < len(tokens):
            print(f"掩盖位置 {pos}: '{tokens[pos]}'")
            mlm_labels[pos] = token_ids[pos]  # 保存原始ID用于损失计算
            masked_tokens[pos] = '[MASK]'
            masked_ids[pos] = tokenizer.get('[MASK]', 2)
    
    print(f"掩码后句子: {' '.join(masked_tokens)}")
    print(f"掩码后 IDs: {masked_ids}")
    print(f"MLM 标签: {mlm_labels}")
    
    return masked_ids, mlm_labels


def demonstrate_nsp_task():
    """演示 NSP 任务"""
    print("\n" + "=" * 60)
    print("下一句预测 (NSP) 任务演示")
    print("=" * 60)
    
    # 正例：连续句子
    sentence_a1 = "人工智能是计算机科学的一个分支"
    sentence_b1 = "它试图理解智能的实质"
    is_next_1 = 1
    
    print(f"正例 (连续句子):")
    print(f"  句子A: {sentence_a1}")
    print(f"  句子B: {sentence_b1}")
    print(f"  标签: {is_next_1} (是下一句)")
    
    # 负例：随机句子
    sentence_a2 = "人工智能是计算机科学的一个分支"
    sentence_b2 = "今天天气很好"
    is_next_2 = 0
    
    print(f"\n负例 (随机句子):")
    print(f"  句子A: {sentence_a2}")
    print(f"  句子B: {sentence_b2}")
    print(f"  标签: {is_next_2} (不是下一句)")
    
    # 构造输入序列
    tokenizer = SimpleTokenizer()
    
    def create_bert_input(sentence_a, sentence_b):
        tokens_a = sentence_a.split()
        tokens_b = sentence_b.split()
        
        # [CLS] sentence_a [SEP] sentence_b [SEP]
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        input_ids = [tokenizer.get(token, 4) for token in tokens]
        
        # Token type IDs
        token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        
        return tokens, input_ids, token_type_ids
    
    print(f"\n正例输入构造:")
    tokens1, ids1, types1 = create_bert_input(sentence_a1, sentence_b1)
    print(f"  Tokens: {tokens1}")
    print(f"  Token类型: {types1}")
    
    print(f"\n负例输入构造:")
    tokens2, ids2, types2 = create_bert_input(sentence_a2, sentence_b2)
    print(f"  Tokens: {tokens2}")
    print(f"  Token类型: {types2}")
    
    return (ids1, types1, is_next_1), (ids2, types2, is_next_2)


if __name__ == "__main__":
    print("🤖 BERT 预训练完整演示")
    print("=" * 60)
    
    # 1. 演示预训练过程
    trainer = demonstrate_bert_pretraining()
    
    # 2. 分析注意力模式
    analyze_bert_attention()
    
    # 3. 演示 MLM 任务
    demonstrate_mlm_task()
    
    # 4. 演示 NSP 任务
    demonstrate_nsp_task()
    
    print("\n" + "=" * 60)
    print("BERT 预训练演示完成！")
    print("✅ 已完成 MLM 和 NSP 预训练任务")
    print("✅ 已分析注意力机制行为")
    print("✅ 已演示数据处理流程")
    print("=" * 60)
