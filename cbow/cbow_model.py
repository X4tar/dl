import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import Counter, defaultdict
import random
import re


class CBOWModel(nn.Module):
    """
    Continuous Bag of Words (CBOW) Model
    
    CBOW预测中心词基于上下文词汇
    """
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOWModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        
        # 词嵌入层 - 将词汇索引映射到向量
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # 线性层 - 从嵌入维度映射到词汇表大小
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化模型权重"""
        # todo
        nn.init.uniform_(self.embeddings.weight, -1.0, 1.0)
        nn.init.uniform_(self.linear.weight, -1.0, 1.0)
        nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, context_words):
        """
        前向传播
        
        Args:
            context_words: 上下文词汇的索引张量 [batch_size, context_size]
        
        Returns:
            输出logits [batch_size, vocab_size]
        """
        # 获取上下文词汇的嵌入 [batch_size, context_size, embedding_dim]
        embeds = self.embeddings(context_words)
        
        # 对上下文词汇的嵌入求平均 [batch_size, embedding_dim]
        context_vector = torch.mean(embeds, dim=1)
        
        # 通过线性层得到输出 [batch_size, vocab_size]
        output = self.linear(context_vector)
        
        return output


class CBOWTrainer:
    """CBOW模型训练器"""
    
    def __init__(self, embedding_dim=100, context_size=2, min_count=2, learning_rate=0.01):
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.min_count = min_count
        self.learning_rate = learning_rate
        
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        self.model = None
        self.optimizer = None
        
    def preprocess_text(self, text):
        """预处理文本数据"""
        # 转换为小写并移除标点符号
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()
        return words
    
    def build_vocabulary(self, corpus):
        """构建词汇表"""
        word_counts = Counter()
        
        for sentence in corpus:
            words = self.preprocess_text(sentence)
            word_counts.update(words)
        
        # 过滤低频词汇
        filtered_words = [word for word, count in word_counts.items() if count >= self.min_count]
        
        # 构建词汇索引映射
        self.word_to_idx = {word: idx for idx, word in enumerate(filtered_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        
        print(f"词汇表大小: {self.vocab_size}")
        
    def create_training_data(self, corpus):
        """创建训练数据"""
        training_data = []
        
        for sentence in corpus:
            words = self.preprocess_text(sentence)
            # 过滤不在词汇表中的词汇
            words = [word for word in words if word in self.word_to_idx]
            
            if len(words) < 2 * self.context_size + 1:
                continue
                
            for i in range(self.context_size, len(words) - self.context_size):
                # 获取上下文词汇
                context = []
                for j in range(i - self.context_size, i + self.context_size + 1):
                    if j != i:  # 排除目标词
                        context.append(self.word_to_idx[words[j]])
                
                # 目标词
                target = self.word_to_idx[words[i]]
                
                training_data.append((context, target))
        
        return training_data
    
    def train(self, corpus, epochs=100, batch_size=64):
        """训练CBOW模型"""
        # 构建词汇表
        self.build_vocabulary(corpus)
        
        # 创建训练数据
        training_data = self.create_training_data(corpus)
        print(f"训练样本数量: {len(training_data)}")
        
        # 初始化模型
        self.model = CBOWModel(self.vocab_size, self.embedding_dim, 2 * self.context_size)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(training_data)
            
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                
                # 准备批次数据
                contexts = torch.tensor([item[0] for item in batch], dtype=torch.long)
                targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
                
                # 前向传播
                self.optimizer.zero_grad()
                logits = self.model(contexts)
                loss = criterion(logits, targets)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(training_data)
                print(f'Epoch {epoch + 1}/{epochs}, 平均损失: {avg_loss:.4f}')
    
    def get_word_embedding(self, word):
        """获取指定词汇的嵌入向量"""
        if not self.model or word not in self.word_to_idx:
            return None
        
        word_idx = self.word_to_idx[word]
        with torch.no_grad():
            embedding = self.model.embeddings.weight[word_idx].detach().numpy()
        return embedding
    
    def find_similar_words(self, word, top_k=5):
        """找到最相似的词汇"""
        if word not in self.word_to_idx:
            return []
        
        target_embedding = self.get_word_embedding(word)
        if target_embedding is None:
            return []
            
        similarities = []
        
        for other_word in self.word_to_idx:
            if other_word != word:
                other_embedding = self.get_word_embedding(other_word)
                if other_embedding is not None:
                    # 计算余弦相似度
                    similarity = np.dot(target_embedding, other_embedding) / (
                        np.linalg.norm(target_embedding) * np.linalg.norm(other_embedding)
                    )
                    similarities.append((other_word, similarity))
        
        # 排序并返回前k个
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def predict_word(self, context_words):
        """根据上下文词汇预测目标词"""
        if self.model is None:
            print("模型未训练!")
            return None
        
        # 转换上下文词汇为索引
        context_indices = []
        for word in context_words:
            if word in self.word_to_idx:
                context_indices.append(self.word_to_idx[word])
            else:
                print(f"词汇 '{word}' 不在词汇表中")
                return None
        
        # 确保上下文长度正确
        if len(context_indices) != 2 * self.context_size:
            print(f"上下文词汇数量应为 {2 * self.context_size}")
            return None
        
        with torch.no_grad():
            context_tensor = torch.tensor([context_indices], dtype=torch.long)
            logits = self.model(context_tensor)
            predicted_idx = int(torch.argmax(logits, dim=1).item())
            predicted_word = self.idx_to_word[predicted_idx]
        
        return predicted_word
    
    def save_model(self, filepath):
        """保存模型"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word,
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'context_size': self.context_size
            }, filepath)
            print(f"模型已保存到 {filepath}")
        else:
            print("模型尚未训练，无法保存")
    
    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath)
        
        self.word_to_idx = checkpoint['word_to_idx']
        self.idx_to_word = checkpoint['idx_to_word']
        self.vocab_size = checkpoint['vocab_size']
        self.embedding_dim = checkpoint['embedding_dim']
        self.context_size = checkpoint['context_size']
        
        self.model = CBOWModel(self.vocab_size, self.embedding_dim, 2 * self.context_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"模型已从 {filepath} 加载")


if __name__ == "__main__":
    # 示例语料库
    sample_corpus = [
        "the quick brown fox jumps over the lazy dog",
        "a quick brown dog jumps over a lazy fox",
        "the cat sits on the mat",
        "a cat runs in the garden",
        "dogs and cats are pets",
        "the sun rises in the east",
        "birds fly in the sky",
        "fish swim in the water",
        "the moon shines at night",
        "stars twinkle in the dark sky"
    ]
    
    # 训练模型
    trainer = CBOWTrainer(embedding_dim=50, context_size=2, learning_rate=0.1)
    trainer.train(sample_corpus, epochs=50)
    
    # 测试词汇相似度
    print("\n=== 词汇相似度测试 ===")
    similar_words = trainer.find_similar_words("cat", top_k=3)
    print(f"与 'cat' 最相似的词汇: {similar_words}")
    
    # 测试词汇预测
    print("\n=== 词汇预测测试 ===")
    context = ["the", "brown", "jumps", "over"]  # 上下文: "the brown _ jumps over"
    predicted = trainer.predict_word(context)
    print(f"上下文 {context} 预测的词汇: {predicted}")
    
    # 保存模型
    trainer.save_model("cbow_model.pth")
