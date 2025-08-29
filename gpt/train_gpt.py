"""
GPT 模型训练脚本
包含数据处理、训练循环、评估等完整流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import time
import os
from typing import List, Tuple
import matplotlib.pyplot as plt

from gpt_model import GPTLMHeadModel, GPTConfig, create_gpt_model

class SimpleTextDataset(Dataset):
    """
    简单的文本数据集
    
    将文本分割成固定长度的序列用于语言建模训练
    """
    
    def __init__(self, text: str, tokenizer, seq_length: int = 128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # 简单的字符级分词
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # 将文本转换为token ids
        self.tokens = [self.char_to_idx[ch] for ch in text]
        
        # 创建训练样本
        self.examples = []
        for i in range(0, len(self.tokens) - seq_length, seq_length):
            input_ids = self.tokens[i:i + seq_length]
            target_ids = self.tokens[i + 1:i + seq_length + 1]
            self.examples.append((input_ids, target_ids))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        input_ids, target_ids = self.examples[idx]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)
    
    @property
    def vocab_size(self):
        return len(self.chars)

class BPETokenizer:
    """
    简化的 BPE 分词器（用于演示）
    实际应用中建议使用 Hugging Face tokenizers
    """
    
    def __init__(self):
        # 基础字符集
        self.vocab = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-\n")
        self.char_to_id = {char: i for i, char in enumerate(self.vocab)}
        self.id_to_char = {i: char for i, char in enumerate(self.vocab)}
        
    def encode(self, text: str) -> List[int]:
        """编码文本为token ids"""
        return [self.char_to_id.get(char, 0) for char in text]
    
    def decode(self, token_ids: List[int]) -> str:
        """解码token ids为文本"""
        return ''.join([self.id_to_char.get(id, '<UNK>') for id in token_ids])
    
    @property
    def vocab_size(self):
        return len(self.vocab)

def create_sample_data():
    """创建示例训练数据"""
    sample_text = """
    Once upon a time, in a land far away, there lived a brave knight named Arthur.
    Arthur was known throughout the kingdom for his courage and wisdom.
    One day, a mysterious dragon appeared and threatened the peaceful village.
    The villagers were scared and didn't know what to do.
    Arthur decided to face the dragon and protect his people.
    He took his sword and shield and went to meet the beast.
    After a fierce battle, Arthur defeated the dragon and saved the village.
    The people celebrated and Arthur became a legend.
    From that day forward, peace returned to the land.
    And they all lived happily ever after.
    
    The end of this simple story shows how courage can overcome fear.
    Sometimes we must be brave to protect what we love.
    Arthur's story teaches us about heroism and selflessness.
    Every hero starts as an ordinary person who chooses to do extraordinary things.
    """
    
    return sample_text.strip()

def calculate_perplexity(loss):
    """计算困惑度"""
    return math.exp(loss)

class GPTTrainer:
    """GPT 训练器"""
    
    def __init__(
        self, 
        model: GPTLMHeadModel,
        tokenizer,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # 训练历史
        self.train_losses = []
        self.train_perplexities = []
        
    def train_epoch(self, dataloader, optimizer, scheduler=None):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # 前向传播
            logits, loss, _ = self.model(input_ids, labels=target_ids)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 打印进度
            if batch_idx % 10 == 0:
                perplexity = calculate_perplexity(loss.item())
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, Perplexity: {perplexity:.2f}')
        
        avg_loss = total_loss / num_batches
        avg_perplexity = calculate_perplexity(avg_loss)
        
        self.train_losses.append(avg_loss)
        self.train_perplexities.append(avg_perplexity)
        
        return avg_loss, avg_perplexity
    
    def evaluate(self, dataloader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, target_ids in dataloader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                logits, loss, _ = self.model(input_ids, labels=target_ids)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_perplexity = calculate_perplexity(avg_loss)
        
        return avg_loss, avg_perplexity
    
    def generate_sample(self, prompt: str, max_length: int = 100, temperature: float = 0.8):
        """生成文本样本"""
        self.model.eval()
        
        # 编码提示
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long).to(self.device)
        
        # 生成
        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_k=50
            )
        
        # 解码
        generated_text = self.tokenizer.decode(generated[0].cpu().tolist())
        return generated_text
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(self.train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # 困惑度曲线
        ax2.plot(self.train_perplexities)
        ax2.set_title('Training Perplexity')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Perplexity')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('gpt/training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_optimizer_and_scheduler(model, num_training_steps, learning_rate=1e-4):
    """创建优化器和学习率调度器"""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )
    
    # Warmup + Cosine Annealing
    warmup_steps = num_training_steps // 10
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (num_training_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler

def main():
    """主训练函数"""
    print("🚀 开始 GPT 训练演示")
    print("=" * 50)
    
    # 设备检查
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建分词器
    tokenizer = BPETokenizer()
    print(f"词汇表大小: {tokenizer.vocab_size}")
    
    # 创建模型
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=256,
        n_embd=128,
        n_layer=6,
        n_head=8,
        n_inner=512
    )
    
    model = GPTLMHeadModel(config)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 准备数据
    print("\n准备训练数据...")
    text_data = create_sample_data()
    
    # 创建数据集
    dataset = SimpleTextDataset(text_data, tokenizer, seq_length=64)
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    
    # 创建训练器
    trainer = GPTTrainer(model, tokenizer, device)
    
    # 创建优化器和调度器
    num_epochs = 20
    num_training_steps = len(train_dataloader) * num_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, num_training_steps)
    
    # 训练循环
    print("\n开始训练...")
    print("=" * 50)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)
        
        # 训练
        start_time = time.time()
        train_loss, train_perplexity = trainer.train_epoch(train_dataloader, optimizer, scheduler)
        
        # 验证
        val_loss, val_perplexity = trainer.evaluate(val_dataloader)
        
        epoch_time = time.time() - start_time
        
        print(f"训练损失: {train_loss:.4f}, 训练困惑度: {train_perplexity:.2f}")
        print(f"验证损失: {val_loss:.4f}, 验证困惑度: {val_perplexity:.2f}")
        print(f"耗时: {epoch_time:.2f}s")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'gpt/best_model.pth')
            print("💾 保存最佳模型")
        
        # 生成样本
        if (epoch + 1) % 5 == 0:
            print("\n📝 生成文本样本:")
            sample = trainer.generate_sample("Once upon a time", max_length=100, temperature=0.8)
            print(f"'{sample}'")
    
    print("\n✅ 训练完成!")
    
    # 绘制训练曲线
    print("📊 绘制训练曲线...")
    trainer.plot_training_curves()
    
    # 最终测试
    print("\n🧪 最终测试:")
    prompts = [
        "Once upon a time",
        "Arthur was",
        "The dragon",
        "Peace returned"
    ]
    
    for prompt in prompts:
        generated = trainer.generate_sample(prompt, max_length=50, temperature=0.7)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated}'")
        print("-" * 40)

if __name__ == "__main__":
    main()
