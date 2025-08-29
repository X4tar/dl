"""
监督微调 (Supervised Fine-tuning, SFT) 训练器
实现指令-回答对的监督学习训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import math
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# 假设我们使用之前实现的 GPT 模型作为基础
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gpt'))

try:
    from gpt_model import GPTLMHeadModel, GPTConfig
except ImportError:
    print("注意：无法导入 GPT 模型，将使用简化实现")

class InstructionDataset(Dataset):
    """
    指令微调数据集
    
    处理指令-回答对数据，支持多种格式
    """
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer, 
        max_length: int = 512,
        instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_template = instruction_template
        
        # 加载数据
        self.data = self.load_data(data_path)
        
        print(f"加载了 {len(self.data)} 条指令数据")
    
    def load_data(self, data_path: str) -> List[Dict]:
        """加载指令数据"""
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # 如果没有真实数据文件，创建示例数据
            data = self.create_sample_data()
        
        return data
    
    def create_sample_data(self) -> List[Dict]:
        """创建示例指令数据"""
        sample_data = [
            {
                "instruction": "请解释什么是人工智能",
                "response": "人工智能(AI)是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这包括学习、推理、感知、理解自然语言、识别模式等能力。"
            },
            {
                "instruction": "用简单的话解释机器学习",
                "response": "机器学习是让计算机从数据中自动学习规律的技术。就像人类通过经验学习一样，机器学习算法通过分析大量数据来发现模式，并用这些模式来对新数据做出预测或决策。"
            },
            {
                "instruction": "写一首关于春天的短诗",
                "response": "春风轻拂绿柳梢，\n花开满树竞妖娆。\n蝶舞蜂飞忙采蜜，\n大地重现好春朝。"
            },
            {
                "instruction": "如何学习编程？",
                "response": "学习编程的建议步骤：1. 选择一门适合初学者的语言，如Python；2. 掌握基础语法和概念；3. 通过实际项目练习；4. 阅读他人的代码学习；5. 持续练习和解决问题；6. 加入编程社区交流学习。"
            },
            {
                "instruction": "解释深度学习和传统机器学习的区别",
                "response": "主要区别：1. 特征提取：深度学习自动学习特征，传统ML需要手工设计；2. 数据需求：深度学习需要大量数据，传统ML在小数据上也能工作；3. 计算资源：深度学习需要更多计算力；4. 可解释性：传统ML更容易解释，深度学习较为黑盒；5. 适用场景：深度学习在图像、语音、NLP等复杂任务上表现更好。"
            }
        ]
        
        # 扩展数据集
        extended_data = []
        for _ in range(20):  # 复制数据以增加训练样本
            extended_data.extend(sample_data)
        
        return extended_data
    
    def format_instruction(self, instruction: str, response: str) -> str:
        """格式化指令-回答对"""
        return self.instruction_template.format(
            instruction=instruction,
            response=response
        )
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # 格式化文本
        formatted_text = self.format_instruction(
            item["instruction"], 
            item["response"]
        )
        
        # 分词
        if hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(formatted_text)
        else:
            # 简单的字符级分词
            tokens = [ord(c) % 256 for c in formatted_text]
        
        # 截断或填充到固定长度
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens.extend([0] * (self.max_length - len(tokens)))
        
        # 创建标签（向右偏移一位）
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'instruction': item["instruction"],
            'response': item["response"]
        }

class SimpleTokenizer:
    """简单的字符级分词器"""
    
    def __init__(self):
        # 支持的字符集
        self.chars = []
        # 添加基本ASCII字符
        for i in range(256):
            self.chars.append(chr(i))
        
        self.char_to_id = {char: i for i, char in enumerate(self.chars)}
        self.id_to_char = {i: char for i, char in enumerate(self.chars)}
    
    def encode(self, text: str) -> List[int]:
        """编码文本"""
        return [self.char_to_id.get(c, 0) for c in text]
    
    def decode(self, token_ids: List[int]) -> str:
        """解码token ids"""
        return ''.join([self.id_to_char.get(id, '') for id in token_ids])
    
    @property
    def vocab_size(self):
        return len(self.chars)

class SFTTrainer:
    """监督微调训练器"""
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cpu',
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # 训练历史
        self.train_losses = []
        self.learning_rates = []
        
    def compute_loss(self, batch):
        """计算损失"""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # 前向传播
        if hasattr(self.model, 'forward'):
            outputs = self.model(input_ids, labels=labels)
            if isinstance(outputs, tuple):
                loss = outputs[1]  # (logits, loss, ...)
            else:
                # 手动计算损失
                logits = outputs
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
        else:
            # 简化的损失计算
            loss = torch.tensor(0.5, requires_grad=True)
        
        return loss
    
    def train_epoch(self, dataloader, epoch: int):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        print(f"\n📚 Epoch {epoch + 1} 开始训练")
        print("-" * 50)
        
        for batch_idx, batch in enumerate(dataloader):
            # 计算损失
            loss = self.compute_loss(batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 参数更新
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 打印进度
            if batch_idx % 5 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx:3d}/{num_batches} | "
                      f"Loss: {loss.item():.4f} | "
                      f"LR: {current_lr:.2e}")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        print(f"\n📊 Epoch {epoch + 1} 完成:")
        print(f"   平均损失: {avg_loss:.4f}")
        print(f"   困惑度: {math.exp(avg_loss):.2f}")
        
        return avg_loss
    
    def evaluate(self, dataloader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        perplexity = math.exp(avg_loss)
        
        return avg_loss, perplexity
    
    def generate_response(self, instruction: str, max_length: int = 100, temperature: float = 0.7):
        """根据指令生成回答"""
        self.model.eval()
        
        # 格式化输入
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        # 编码
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long).to(self.device)
        
        # 生成（如果模型支持）
        if hasattr(self.model, 'generate'):
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_length=len(input_ids[0]) + max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50
                )
            
            # 解码
            generated_text = self.tokenizer.decode(generated[0].cpu().tolist())
            
            # 提取回答部分
            if "### Response:\n" in generated_text:
                response = generated_text.split("### Response:\n")[-1]
            else:
                response = generated_text[len(prompt):]
            
            return response.strip()
        else:
            return "模型不支持生成功能"
    
    def save_model(self, save_path: str):
        """保存模型"""
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
        }, save_path)
        
        print(f"💾 模型已保存到: {save_path}")
    
    def load_model(self, load_path: str):
        """加载模型"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        
        print(f"📂 模型已从 {load_path} 加载")

def create_sample_model(vocab_size: int = 256):
    """创建示例模型用于演示"""
    try:
        # 尝试使用 GPT 模型
        config = GPTConfig(
            vocab_size=vocab_size,
            n_positions=512,
            n_embd=256,
            n_layer=6,
            n_head=8,
            n_inner=1024
        )
        model = GPTLMHeadModel(config)
        print(f"✅ 使用 GPT 模型 ({sum(p.numel() for p in model.parameters()):,} 参数)")
        return model
    except:
        # 简化的模型实现
        class SimpleModel(nn.Module):
            def __init__(self, vocab_size, embed_dim=256):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.transformer = nn.TransformerDecoderLayer(
                    d_model=embed_dim,
                    nhead=8,
                    batch_first=True
                )
                self.lm_head = nn.Linear(embed_dim, vocab_size)
                
            def forward(self, input_ids, labels=None):
                x = self.embedding(input_ids)
                x = self.transformer(x, x)  # 简化版本
                logits = self.lm_head(x)
                
                loss = None
                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )
                
                return logits, loss
        
        model = SimpleModel(vocab_size)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ 使用简化模型 ({param_count:,} 参数)")
        return model

def main():
    """主函数 - 演示监督微调流程"""
    print("🎯 监督微调 (SFT) 训练演示")
    print("=" * 50)
    
    # 设备设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 使用设备: {device}")
    
    # 创建分词器
    tokenizer = SimpleTokenizer()
    print(f"📝 词汇表大小: {tokenizer.vocab_size}")
    
    # 创建模型
    model = create_sample_model(tokenizer.vocab_size)
    
    # 创建数据集
    dataset = InstructionDataset(
        data_path="dummy_path",
        tokenizer=tokenizer,
        max_length=256
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    # 创建训练器
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=1e-4
    )
    
    # 训练演示
    print(f"\n🚀 开始训练 {len(dataset)} 个样本...")
    
    num_epochs = 3
    for epoch in range(num_epochs):
        avg_loss = trainer.train_epoch(dataloader, epoch)
        
        # 每个epoch后生成示例
        if epoch % 1 == 0:
            print(f"\n🤖 第 {epoch + 1} 轮训练后的生成示例:")
            test_instruction = "请解释什么是深度学习"
            response = trainer.generate_response(test_instruction, max_length=50)
            print(f"   指令: {test_instruction}")
            print(f"   回答: {response}")
    
    # 保存模型
    save_path = "instruction_tuning/sft_model.pth"
    trainer.save_model(save_path)
    
    print(f"\n✅ 监督微调训练完成！")
    print(f"📈 训练损失变化: {trainer.train_losses}")

if __name__ == "__main__":
    main()
