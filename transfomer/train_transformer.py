"""
Transformer 训练示例
包含完整的训练流程，从数据准备到模型训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import random
import time
from transformer_model import Transformer, TransformerForLanguageModeling
from transformer_components import create_padding_mask, create_look_ahead_mask

class SimpleTextDataset(Dataset):
    """
    简单的文本数据集
    用于机器翻译任务的示例数据集
    """
    
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, max_len=50):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        
        # 特殊标记
        self.PAD = 0
        self.BOS = 1  # 开始标记
        self.EOS = 2  # 结束标记
        self.UNK = 3  # 未知标记
        
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        # 转换为token IDs
        src_ids = self.text_to_ids(src_text, self.src_vocab)
        tgt_ids = self.text_to_ids(tgt_text, self.tgt_vocab)
        
        # 添加BOS和EOS标记
        tgt_input = [self.BOS] + tgt_ids[:-1]  # 解码器输入
        tgt_output = tgt_ids + [self.EOS]      # 解码器目标
        
        # 填充到固定长度
        src_ids = self.pad_sequence(src_ids, self.max_len)
        tgt_input = self.pad_sequence(tgt_input, self.max_len)
        tgt_output = self.pad_sequence(tgt_output, self.max_len)
        
        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt_input': torch.tensor(tgt_input, dtype=torch.long),
            'tgt_output': torch.tensor(tgt_output, dtype=torch.long)
        }
    
    def text_to_ids(self, text, vocab):
        """将文本转换为ID序列"""
        words = text.lower().split()
        return [vocab.get(word, self.UNK) for word in words]
    
    def pad_sequence(self, seq, max_len):
        """填充序列到指定长度"""
        if len(seq) >= max_len:
            return seq[:max_len]
        else:
            return seq + [self.PAD] * (max_len - len(seq))


def build_vocab(texts, min_freq=2):
    """
    构建词汇表
    
    Args:
        texts: 文本列表
        min_freq: 最小频率阈值
        
    Returns:
        vocab: 词汇表字典 {word: id}
        id2word: 反向词汇表 {id: word}
    """
    # 统计词频
    word_counter = Counter()
    for text in texts:
        words = text.lower().split()
        word_counter.update(words)
    
    # 构建词汇表
    vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for word, freq in word_counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    # 反向词汇表
    id2word = {id: word for word, id in vocab.items()}
    
    return vocab, id2word


def create_sample_data():
    """
    创建简单的翻译数据样本
    英语 -> 法语 （简化版本）
    """
    # 英语句子
    english_sentences = [
        "hello world",
        "how are you",
        "good morning",
        "thank you very much",
        "have a good day",
        "see you later",
        "nice to meet you",
        "what is your name",
        "where are you from",
        "i love you",
        "this is a book",
        "the cat is sleeping",
        "i want to learn",
        "it is raining today",
        "the sun is shining",
        "i am hungry",
        "let us go home",
        "the flowers are beautiful",
        "music makes me happy",
        "tomorrow will be better",
        "hello my friend",
        "how old are you",
        "good night sleep well",
        "thank you for help",
        "have a nice weekend",
        "see you tomorrow morning",
        "nice to see you again",
        "what time is it now",
        "where do you live now",
        "i love this song very much",
        "this is my favorite book",
        "the dog is running fast",
        "i want to learn english",
        "it is sunny today morning",
        "the moon is bright tonight",
        "i am very tired now",
        "let us go to school",
        "the garden flowers are blooming",
        "classical music makes me relaxed",
        "tomorrow will be much better"
    ]
    
    # 对应的法语句子（简化）
    french_sentences = [
        "bonjour monde",
        "comment allez vous",
        "bonjour matin",
        "merci beaucoup bien",
        "passez bonne journee",
        "voir plus tard",
        "ravi rencontrer vous",
        "quel est votre nom",
        "ou etes vous de",
        "je vous aime",
        "ceci est livre",
        "chat dort maintenant",
        "je veux apprendre",
        "il pleut aujourd hui",
        "soleil brille maintenant",
        "je suis faim",
        "allons maison maintenant",
        "fleurs sont belles",
        "musique rend heureux",
        "demain sera meilleur",
        "bonjour mon ami",
        "quel age avez vous",
        "bonne nuit dormez bien",
        "merci pour aide donnee",
        "passez bon weekend agréable",
        "voir demain matin tot",
        "ravi vous revoir encore",
        "quelle heure est maintenant",
        "ou habitez vous maintenant",
        "je aime cette chanson beaucoup",
        "ceci est mon livre prefere",
        "chien court tres rapidement",
        "je veux apprendre anglais langue",
        "il fait soleil aujourd hui matin",
        "lune est brillante ce soir",
        "je suis tres fatigue maintenant",
        "allons ecole ensemble maintenant",
        "jardin fleurs sont en fleur",
        "musique classique rend detendu paisible",
        "demain sera beaucoup mieux certainement"
    ]
    
    return english_sentences, french_sentences


class TransformerTrainer:
    """Transformer 训练器"""
    
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=0.0001, warmup_steps=4000):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 优化器 - 使用Adam with warmup
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
                                  betas=(0.9, 0.98), eps=1e-9)
        
        # 学习率调度器
        self.warmup_steps = warmup_steps
        self.d_model = model.d_model
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
    def get_lr(self, step):
        """获取当前学习率（带warmup）"""
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        return (self.d_model ** -0.5) * min(arg1, arg2)
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            src = batch['src'].to(self.device)
            tgt_input = batch['tgt_input'].to(self.device)
            tgt_output = batch['tgt_output'].to(self.device)
            
            # 创建掩码
            src_mask = create_padding_mask(src).to(self.device)
            
            # 创建目标掩码（因果掩码 + 填充掩码）
            tgt_seq_len = tgt_input.size(1)
            tgt_mask = create_look_ahead_mask(tgt_seq_len).to(self.device)
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
            tgt_padding_mask = create_padding_mask(tgt_input).to(self.device)
            tgt_mask = tgt_mask & tgt_padding_mask
            
            # 前向传播
            output, _, _, _ = self.model(src, tgt_input, src_mask, tgt_mask)
            
            # 计算损失
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
            loss = self.criterion(output, tgt_output)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 更新学习率
            step = len(self.train_losses) * num_batches + batch_idx + 1
            lr = self.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}, LR: {lr:.6f}')
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                src = batch['src'].to(self.device)
                tgt_input = batch['tgt_input'].to(self.device)
                tgt_output = batch['tgt_output'].to(self.device)
                
                # 创建掩码
                src_mask = create_padding_mask(src).to(self.device)
                tgt_seq_len = tgt_input.size(1)
                tgt_mask = create_look_ahead_mask(tgt_seq_len).to(self.device)
                tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
                tgt_padding_mask = create_padding_mask(tgt_input).to(self.device)
                tgt_mask = tgt_mask & tgt_padding_mask
                
                # 前向传播
                output, _, _, _ = self.model(src, tgt_input, src_mask, tgt_mask)
                
                # 计算损失
                output = output.reshape(-1, output.size(-1))
                tgt_output = tgt_output.reshape(-1)
                loss = self.criterion(output, tgt_output)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, num_epochs):
        """完整训练流程"""
        print(f"开始训练 {num_epochs} 个epochs...")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 记录时间
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  训练损失: {train_loss:.4f}')
            print(f'  验证损失: {val_loss:.4f}')
            print(f'  用时: {epoch_time:.2f}秒')
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'transfomer/best_transformer_model.pth')
                print(f'  保存最佳模型 (验证损失: {val_loss:.4f})')
            
            print('-' * 50)
    
    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title('训练和验证损失')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失 (对数尺度)')
        plt.title('训练和验证损失 (对数尺度)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('transfomer/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


def translate_sentence(model, sentence, src_vocab, tgt_vocab, tgt_id2word, 
                      device, max_len=50):
    """
    翻译单个句子
    
    Args:
        model: 训练好的模型
        sentence: 输入句子
        src_vocab: 源语言词汇表
        tgt_vocab: 目标语言词汇表
        tgt_id2word: 目标语言反向词汇表
        device: 设备
        max_len: 最大长度
        
    Returns:
        translated: 翻译结果
    """
    model.eval()
    
    # 预处理输入句子
    words = sentence.lower().split()
    src_ids = [src_vocab.get(word, 3) for word in words]  # 3是UNK标记
    
    # 转换为tensor
    src = torch.tensor([src_ids], dtype=torch.long).to(device)
    src_mask = create_padding_mask(src).to(device)
    
    # 编码
    encoder_output, _ = model.encode(src, src_mask)
    
    # 解码
    tgt_ids = [1]  # 从BOS开始
    
    for _ in range(max_len):
        tgt = torch.tensor([tgt_ids], dtype=torch.long).to(device)
        tgt_seq_len = len(tgt_ids)
        tgt_mask = create_look_ahead_mask(tgt_seq_len).to(device)
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
        
        # 解码一步
        with torch.no_grad():
            output, _, _ = model.decode(tgt, encoder_output, src_mask, tgt_mask)
            next_token = output[0, -1].argmax().item()
        
        tgt_ids.append(next_token)
        
        # 如果生成了EOS，停止
        if next_token == 2:  # EOS标记
            break
    
    # 转换为文本
    translated_words = []
    for token_id in tgt_ids[1:-1]:  # 跳过BOS和EOS
        if token_id in tgt_id2word:
            translated_words.append(tgt_id2word[token_id])
        else:
            translated_words.append('<UNK>')
    
    return ' '.join(translated_words)


def main():
    """主训练函数"""
    print("=" * 60)
    print("Transformer 训练示例")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据
    print("\n1. 创建训练数据...")
    english_sentences, french_sentences = create_sample_data()
    
    # 分割训练和验证集
    split_idx = int(0.8 * len(english_sentences))
    train_en = english_sentences[:split_idx]
    train_fr = french_sentences[:split_idx]
    val_en = english_sentences[split_idx:]
    val_fr = french_sentences[split_idx:]
    
    print(f"训练集大小: {len(train_en)}")
    print(f"验证集大小: {len(val_en)}")
    
    # 构建词汇表
    print("\n2. 构建词汇表...")
    src_vocab, src_id2word = build_vocab(english_sentences, min_freq=1)
    tgt_vocab, tgt_id2word = build_vocab(french_sentences, min_freq=1)
    
    print(f"英语词汇表大小: {len(src_vocab)}")
    print(f"法语词汇表大小: {len(tgt_vocab)}")
    
    # 创建数据集和数据加载器
    print("\n3. 创建数据加载器...")
    train_dataset = SimpleTextDataset(train_en, train_fr, src_vocab, tgt_vocab)
    val_dataset = SimpleTextDataset(val_en, val_fr, src_vocab, tgt_vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # 创建模型
    print("\n4. 创建模型...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        max_seq_len=100,
        dropout=0.1
    )
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 创建训练器
    print("\n5. 开始训练...")
    trainer = TransformerTrainer(
        model, train_loader, val_loader, device,
        learning_rate=0.0001, warmup_steps=1000
    )
    
    # 训练模型
    trainer.train(num_epochs=20)
    
    # 绘制训练历史
    print("\n6. 绘制训练历史...")
    try:
        trainer.plot_training_history()
    except ImportError:
        print("需要安装 matplotlib 来绘制图表")
    
    # 测试翻译
    print("\n7. 测试翻译...")
    test_sentences = [
        "hello world",
        "how are you",
        "good morning",
        "thank you very much"
    ]
    
    for sentence in test_sentences:
        translation = translate_sentence(
            model, sentence, src_vocab, tgt_vocab, tgt_id2word, device
        )
        print(f"英语: {sentence}")
        print(f"法语: {translation}")
        print("-" * 30)
    
    print("\n训练完成！")
    print("模型已保存到 'transfomer/best_transformer_model.pth'")


if __name__ == "__main__":
    main()
