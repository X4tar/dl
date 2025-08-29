"""
CBOW模型使用示例
演示如何使用CBOW模型进行词嵌入学习和词汇预测
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from cbow_model import CBOWTrainer


def create_larger_corpus():
    """创建更大的示例语料库"""
    corpus = [
        # 动物相关
        "the cat sits on the mat and sleeps peacefully",
        "a dog runs quickly in the green park",
        "birds fly high in the blue sky during summer",
        "fish swim deep in the clear water",
        "the horse gallops fast across the open field",
        "cats and dogs are common household pets",
        "the elephant is a large gray animal",
        "lions hunt in groups during the night",
        
        # 自然相关
        "the sun rises early in the morning",
        "the moon shines bright at night",
        "stars twinkle beautifully in the dark sky",
        "rain falls gently on the green leaves",
        "wind blows softly through the tall trees",
        "snow covers the mountain peaks in winter",
        "flowers bloom colorfully in the spring garden",
        "rivers flow smoothly toward the ocean",
        
        # 日常生活
        "people walk slowly in the busy street",
        "children play happily in the school playground",
        "students study hard for their important exams",
        "teachers explain lessons clearly to their students",
        "cars drive fast on the highway",
        "buses stop frequently at various stations",
        "books contain knowledge and interesting stories",
        "computers help people work more efficiently",
        
        # 食物相关
        "apples taste sweet and are very healthy",
        "bread is baked fresh every morning",
        "coffee smells good in the early morning",
        "water is essential for all living things",
        "vegetables grow well in fertile soil",
        "fruits provide vitamins and natural sugar",
        
        # 时间和空间
        "time passes quickly when you are busy",
        "distance seems shorter when traveling by plane",
        "morning light enters through the window",
        "evening shadows grow longer and darker",
        "yesterday was cold but today is warm",
        "tomorrow will bring new opportunities and challenges"
    ]
    return corpus


def visualize_embeddings(trainer, words_to_plot=None):
    """可视化词嵌入"""
    if words_to_plot is None:
        # 选择一些有趣的词汇进行可视化
        words_to_plot = ['cat', 'dog', 'bird', 'fish', 'sun', 'moon', 'water', 'sky',
                        'morning', 'night', 'fast', 'slow', 'good', 'bright']
    
    # 过滤存在于词汇表中的词汇
    available_words = [word for word in words_to_plot if word in trainer.word_to_idx]
    
    if len(available_words) < 2:
        print("可视化需要至少2个词汇")
        return
    
    # 获取词汇嵌入
    embeddings = []
    labels = []
    
    for word in available_words:
        embedding = trainer.get_word_embedding(word)
        if embedding is not None:
            embeddings.append(embedding)
            labels.append(word)
    
    if len(embeddings) < 2:
        print("有效词汇嵌入不足")
        return
    
    # 使用PCA降维到2D
    embeddings = np.array(embeddings)
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # 绘制散点图
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=100)
    
    # 添加词汇标签
    for i, label in enumerate(labels):
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=12)
    
    plt.title('CBOW词嵌入可视化 (PCA降维)', fontsize=16)
    plt.xlabel(f'PC1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    plt.ylabel(f'PC2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cbow_embeddings_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def test_word_analogies(trainer):
    """测试词汇类比"""
    print("\n=== 词汇类比测试 ===")
    
    def analogy_test(word_a, word_b, word_c, trainer):
        """执行词汇类比测试: A 对于 B，就像 C 对于 ?"""
        # 获取词汇嵌入
        try:
            emb_a = trainer.get_word_embedding(word_a)
            emb_b = trainer.get_word_embedding(word_b)
            emb_c = trainer.get_word_embedding(word_c)
            
            if emb_a is None or emb_b is None or emb_c is None:
                return None
            
            # 计算目标向量: C + (B - A)
            target_vector = emb_c + (emb_b - emb_a)
            
            # 找到最相似的词汇
            best_word = None
            best_similarity = -1
            
            for word in trainer.word_to_idx:
                if word not in [word_a, word_b, word_c]:
                    word_emb = trainer.get_word_embedding(word)
                    similarity = np.dot(target_vector, word_emb) / (
                        np.linalg.norm(target_vector) * np.linalg.norm(word_emb)
                    )
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_word = word
            
            return best_word, best_similarity
        except:
            return None
    
    # 测试一些类比
    analogies = [
        ("cat", "cats", "dog"),  # 单复数类比
        ("day", "night", "sun"),  # 对比类比
        ("fast", "slow", "big"),  # 反义词类比
    ]
    
    for word_a, word_b, word_c in analogies:
        result = analogy_test(word_a, word_b, word_c, trainer)
        if result:
            predicted_word, similarity = result
            print(f"{word_a} : {word_b} = {word_c} : {predicted_word} (相似度: {similarity:.3f})")
        else:
            print(f"无法执行类比 {word_a} : {word_b} = {word_c} : ?")


def interactive_prediction_demo(trainer):
    """交互式预测演示"""
    print("\n=== 交互式词汇预测演示 ===")
    print("输入上下文词汇，模型将预测中间的词汇")
    print(f"需要输入 {2 * trainer.context_size} 个上下文词汇")
    print("输入 'quit' 退出")
    
    while True:
        try:
            user_input = input(f"\n请输入 {2 * trainer.context_size} 个上下文词汇 (用空格分隔): ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            context_words = user_input.split()
            
            if len(context_words) != 2 * trainer.context_size:
                print(f"请输入恰好 {2 * trainer.context_size} 个词汇")
                continue
            
            predicted_word = trainer.predict_word(context_words)
            
            if predicted_word:
                print(f"预测的词汇: '{predicted_word}'")
                
                # 显示概率分布的前5个候选词
                context_indices = [trainer.word_to_idx[word] for word in context_words 
                                 if word in trainer.word_to_idx]
                
                if len(context_indices) == len(context_words):
                    import torch
                    with torch.no_grad():
                        context_tensor = torch.tensor([context_indices], dtype=torch.long)
                        log_probs = trainer.model(context_tensor)
                        probs = torch.exp(log_probs).squeeze()
                        
                        # 获取前5个最可能的词汇
                        top_probs, top_indices = torch.topk(probs, k=min(5, len(probs)))
                        
                        print("前5个候选词汇:")
                        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                            word = trainer.idx_to_word[idx.item()]
                            print(f"  {i+1}. {word}: {prob.item():.4f}")
            else:
                print("预测失败，请检查输入的词汇是否在词汇表中")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"发生错误: {e}")


def comprehensive_evaluation(trainer):
    """综合评估模型性能"""
    print("\n=== 模型综合评估 ===")
    
    # 词汇表统计
    print(f"词汇表大小: {trainer.vocab_size}")
    print(f"嵌入维度: {trainer.embedding_dim}")
    print(f"上下文窗口大小: {trainer.context_size}")
    
    # 词汇频率分析
    from collections import Counter
    word_freq = Counter()
    
    # 这里简化处理，实际应该基于训练数据
    for word in trainer.word_to_idx:
        word_freq[word] = len(trainer.find_similar_words(word, top_k=1))
    
    print(f"\n最常见的10个词汇:")
    for word, freq in word_freq.most_common(10):
        print(f"  {word}: {freq}")
    
    # 相似度测试
    test_words = ['the', 'cat', 'dog', 'sun', 'water']
    available_test_words = [w for w in test_words if w in trainer.word_to_idx]
    
    print(f"\n词汇相似度测试:")
    for word in available_test_words[:3]:  # 限制测试数量
        similar = trainer.find_similar_words(word, top_k=3)
        print(f"  与 '{word}' 最相似的词汇: {[w[0] for w in similar]}")


def main():
    """主函数"""
    print("=== CBOW模型训练和演示 ===\n")
    
    # 创建更大的语料库
    corpus = create_larger_corpus()
    print(f"语料库包含 {len(corpus)} 个句子")
    
    # 初始化和训练模型
    trainer = CBOWTrainer(
        embedding_dim=100, 
        context_size=2, 
        min_count=2, 
        learning_rate=0.05
    )
    
    print("\n开始训练模型...")
    trainer.train(corpus, epochs=100, batch_size=32)
    
    # 综合评估
    comprehensive_evaluation(trainer)
    
    # 词汇类比测试
    test_word_analogies(trainer)
    
    # 可视化词嵌入
    try:
        visualize_embeddings(trainer)
        print("\n词嵌入可视化已保存为 'cbow_embeddings_visualization.png'")
    except Exception as e:
        print(f"可视化失败: {e}")
    
    # 保存模型
    trainer.save_model("trained_cbow_model.pth")
    
    # 交互式演示 (可选)
    user_choice = input("\n是否进行交互式预测演示? (y/n): ").strip().lower()
    if user_choice == 'y' or user_choice == 'yes':
        interactive_prediction_demo(trainer)
    
    print("\n演示完成!")


if __name__ == "__main__":
    main()
