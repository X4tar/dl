"""
文本生成示例
展示如何使用 Transformer 进行文本生成
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import re
from collections import Counter
from transformer_model import TransformerForLanguageModeling
from transformer_components import create_look_ahead_mask

class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(self, text, vocab, seq_len=50):
        self.text = text
        self.vocab = vocab
        self.seq_len = seq_len
        self.id2word = {id: word for word, id in vocab.items()}
        
        # 将文本转换为token序列
        words = text.lower().split()
        self.tokens = [vocab.get(word, vocab['<UNK>']) for word in words]
        
        # 创建训练样本
        self.samples = []
        for i in range(0, len(self.tokens) - seq_len, seq_len//2):
            input_seq = self.tokens[i:i+seq_len]
            target_seq = self.tokens[i+1:i+seq_len+1]
            if len(input_seq) == seq_len and len(target_seq) == seq_len:
                self.samples.append((input_seq, target_seq))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.samples[idx]
        return {
            'input_ids': torch.tensor(input_seq, dtype=torch.long),
            'target_ids': torch.tensor(target_seq, dtype=torch.long)
        }


def create_shakespeare_data():
    """创建莎士比亚风格的简单文本数据"""
    shakespeare_text = """
    to be or not to be that is the question whether tis nobler in the mind to suffer
    the slings and arrows of outrageous fortune or to take arms against a sea of troubles
    and by opposing end them to die to sleep no more and by a sleep to say we end
    the heartache and the thousand natural shocks that flesh is heir to tis a consummation
    devoutly to be wished to die to sleep to sleep perchance to dream ay theres the rub
    for in that sleep of death what dreams may come when we have shuffled off this mortal coil
    must give us pause theres the respect that makes calamity of so long life
    for who would bear the whips and scorns of time the oppressors wrong the proud mans contumely
    the pangs of despised love the laws delay the insolence of office and the spurns
    that patient merit of the unworthy takes when he himself might his quietus make
    with a bare bodkin who would fardels bear to grunt and sweat under a weary life
    but that the dread of something after death the undiscovered country from whose bourn
    no traveller returns puzzles the will and makes us rather bear those ills we have
    than fly to others that we know not of thus conscience does make cowards of us all
    and thus the native hue of resolution is sicklied oer with the pale cast of thought
    and enterprises of great pith and moment with this regard their currents turn awry
    and lose the name of action soft you now the fair ophelia nymph in thy orisons
    be all my sins remembered
    """
    
    # 扩展数据集
    extended_text = shakespeare_text * 3  # 重复3次增加训练数据
    
    return extended_text.strip()


def build_vocab_from_text(text, min_freq=2):
    """从文本构建词汇表"""
    words = text.lower().split()
    word_counter = Counter(words)
    
    # 特殊token
    vocab = {'<PAD>': 0, '<UNK>': 1}
    
    # 添加高频词
    for word, freq in word_counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab


class TextGenerator:
    """文本生成器"""
    
    def __init__(self, model, vocab, device):
        self.model = model
        self.vocab = vocab
        self.id2word = {id: word for word, id in vocab.items()}
        self.device = device
    
    def generate_text(self, prompt, max_length=100, temperature=1.0, top_k=50):
        """
        生成文本
        
        Args:
            prompt: 起始文本
            max_length: 最大生成长度
            temperature: 温度参数，控制随机性
            top_k: top-k采样
            
        Returns:
            generated_text: 生成的文本
        """
        self.model.eval()
        
        # 将prompt转换为token
        words = prompt.lower().split()
        input_ids = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        generated_ids = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # 准备输入
                input_tensor = torch.tensor([generated_ids], dtype=torch.long).to(self.device)
                
                # 前向传播
                logits, _ = self.model(input_tensor)
                
                # 获取最后一个位置的logits
                next_token_logits = logits[0, -1, :] / temperature
                
                # Top-k采样
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    # 创建mask
                    mask = torch.full_like(next_token_logits, float('-inf'))
                    mask[top_k_indices] = top_k_logits
                    next_token_logits = mask
                
                # 采样下一个token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                generated_ids.append(next_token)
                
                # 如果生成了停止词或达到最大长度，停止
                if len(generated_ids) >= max_length + len(input_ids):
                    break
        
        # 转换回文本
        generated_words = []
        for token_id in generated_ids:
            if token_id in self.id2word:
                generated_words.append(self.id2word[token_id])
            else:
                generated_words.append('<UNK>')
        
        return ' '.join(generated_words)
    
    def generate_with_nucleus_sampling(self, prompt, max_length=100, temperature=1.0, top_p=0.9):
        """使用nucleus采样生成文本"""
        self.model.eval()
        
        words = prompt.lower().split()
        input_ids = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        generated_ids = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                input_tensor = torch.tensor([generated_ids], dtype=torch.long).to(self.device)
                logits, _ = self.model(input_tensor)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Nucleus (top-p) 采样
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除累积概率超过top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
                
                # 采样
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                generated_ids.append(next_token)
                
                if len(generated_ids) >= max_length + len(input_ids):
                    break
        
        # 转换回文本
        generated_words = []
        for token_id in generated_ids:
            if token_id in self.id2word:
                generated_words.append(self.id2word[token_id])
        
        return ' '.join(generated_words)


def train_language_model():
    """训练语言模型"""
    print("=" * 60)
    print("文本生成训练示例")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据
    print("\n1. 创建训练数据...")
    text_data = create_shakespeare_data()
    print(f"文本长度: {len(text_data)} 字符")
    print(f"词汇数量: {len(text_data.split())} 词")
    
    # 构建词汇表
    print("\n2. 构建词汇表...")
    vocab = build_vocab_from_text(text_data, min_freq=1)
    print(f"词汇表大小: {len(vocab)}")
    
    # 创建数据集
    print("\n3. 创建数据集...")
    dataset = TextDataset(text_data, vocab, seq_len=32)
    print(f"训练样本数量: {len(dataset)}")
    
    # 数据分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 创建模型
    print("\n4. 创建模型...")
    model = TransformerForLanguageModeling(
        vocab_size=len(vocab),
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        max_seq_len=64,
        dropout=0.1
    ).to(device)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {total_params:,}")
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    print("\n5. 开始训练...")
    num_epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            # 前向传播
            logits, _ = model(input_ids)
            
            # 计算损失
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        
        # 验证模式
        model.eval()
        val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                
                logits, _ = model(input_ids)
                loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
                
                val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  训练损失: {avg_train_loss:.4f}')
        print(f'  验证损失: {avg_val_loss:.4f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'transfomer/language_model.pth')
            print(f'  保存最佳模型')
        
        print('-' * 40)
    
    return model, vocab, device


def test_text_generation():
    """测试文本生成"""
    print("\n6. 测试文本生成...")
    
    # 训练模型
    model, vocab, device = train_language_model()
    
    # 创建文本生成器
    generator = TextGenerator(model, vocab, device)
    
    # 测试提示词
    prompts = [
        "to be or not",
        "the question whether",
        "to die to sleep",
        "what dreams may come"
    ]
    
    print("\n生成结果:")
    print("=" * 60)
    
    for i, prompt in enumerate(prompts):
        print(f"\n提示词 {i+1}: '{prompt}'")
        print("-" * 40)
        
        # 贪心生成
        print("1. 贪心生成 (temperature=0.1):")
        generated_text = generator.generate_text(
            prompt, max_length=30, temperature=0.1, top_k=1
        )
        print(f"   {generated_text}")
        
        # 随机生成
        print("2. 随机生成 (temperature=1.0):")
        generated_text = generator.generate_text(
            prompt, max_length=30, temperature=1.0, top_k=10
        )
        print(f"   {generated_text}")
        
        # Nucleus采样
        print("3. Nucleus采样 (top_p=0.9):")
        generated_text = generator.generate_with_nucleus_sampling(
            prompt, max_length=30, temperature=0.8, top_p=0.9
        )
        print(f"   {generated_text}")
        
        print()


def analyze_generation_quality():
    """分析生成质量"""
    print("分析文本生成质量...")
    
    # 这里可以添加各种评估指标
    # 如困惑度(perplexity)、BLEU分数等
    
    print("质量分析功能待实现...")
    print("可以添加以下评估指标:")
    print("- 困惑度 (Perplexity)")
    print("- BLEU 分数")
    print("- 词汇多样性")
    print("- 语法正确性")


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    try:
        # 测试文本生成
        test_text_generation()
        
        # 分析生成质量
        print("\n" + "=" * 60)
        analyze_generation_quality()
        
    except Exception as e:
        print(f"运行时出错: {e}")
        print("这可能是因为缺少依赖或训练数据不足")
    
    print("\n文本生成示例完成！")
