"""
CBOW模型测试脚本
验证模型的基本功能
"""

import torch
import numpy as np
from cbow_model import CBOWModel, CBOWTrainer


def test_cbow_model_architecture():
    """测试CBOW模型架构"""
    print("=== 测试模型架构 ===")
    
    vocab_size = 100
    embedding_dim = 50
    context_size = 4  # 2*context_size个上下文词
    batch_size = 8
    
    # 创建模型
    model = CBOWModel(vocab_size, embedding_dim, context_size)
    
    # 创建随机输入
    context_input = torch.randint(0, vocab_size, (batch_size, context_size))
    
    # 前向传播
    output = model(context_input)
    
    # 验证输出形状
    expected_shape = (batch_size, vocab_size)
    assert output.shape == expected_shape, f"输出形状错误: {output.shape} != {expected_shape}"
    
    # 验证概率分布 (log_softmax的输出)
    probs = torch.exp(output)
    prob_sums = torch.sum(probs, dim=1)
    assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-5), "概率分布不正确"
    
    print("✓ 模型架构测试通过")


def test_trainer_functionality():
    """测试训练器功能"""
    print("\n=== 测试训练器功能 ===")
    
    # 简单测试语料
    test_corpus = [
        "the cat sits on the mat",
        "a dog runs in the park",
        "cats and dogs are pets",
        "the quick brown fox jumps",
        "birds fly in the sky"
    ]
    
    # 创建训练器
    trainer = CBOWTrainer(embedding_dim=20, context_size=1, min_count=1, learning_rate=0.1)
    
    # 测试词汇表构建
    trainer.build_vocabulary(test_corpus)
    assert trainer.vocab_size > 0, "词汇表构建失败"
    assert 'the' in trainer.word_to_idx, "常见词汇未在词汇表中"
    print(f"✓ 词汇表构建成功，大小: {trainer.vocab_size}")
    
    # 测试训练数据创建
    training_data = trainer.create_training_data(test_corpus)
    assert len(training_data) > 0, "训练数据创建失败"
    print(f"✓ 训练数据创建成功，数量: {len(training_data)}")
    
    # 测试模型训练
    trainer.train(test_corpus, epochs=5, batch_size=4)
    assert trainer.model is not None, "模型训练失败"
    print("✓ 模型训练完成")
    
    # 测试词嵌入获取
    if 'the' in trainer.word_to_idx:
        embedding = trainer.get_word_embedding('the')
        assert embedding is not None, "词嵌入获取失败"
        assert embedding.shape == (trainer.embedding_dim,), "词嵌入维度错误"
        print("✓ 词嵌入获取成功")
    
    # 测试相似词汇查找
    if 'cat' in trainer.word_to_idx:
        similar_words = trainer.find_similar_words('cat', top_k=2)
        assert isinstance(similar_words, list), "相似词汇查找返回类型错误"
        print("✓ 相似词汇查找功能正常")
    
    # 测试词汇预测 (只在有足够词汇时测试)
    if len(trainer.word_to_idx) >= 4:
        context_words = list(trainer.word_to_idx.keys())[:2]  # 取前2个词作为上下文
        if len(context_words) == 2:
            predicted = trainer.predict_word(context_words)
            print(f"✓ 词汇预测功能正常，预测结果: {predicted}")


def test_model_save_load():
    """测试模型保存和加载"""
    print("\n=== 测试模型保存和加载 ===")
    
    # 创建并训练一个简单模型
    corpus = ["the cat sits", "the dog runs", "cats and dogs"]
    trainer1 = CBOWTrainer(embedding_dim=10, context_size=1, min_count=1)
    trainer1.train(corpus, epochs=3)
    
    # 保存模型
    save_path = "test_model.pth"
    trainer1.save_model(save_path)
    
    # 获取原始词嵌入用于比较
    original_embedding = None
    if 'the' in trainer1.word_to_idx:
        embed = trainer1.get_word_embedding('the')
        if embed is not None:
            original_embedding = embed.copy()
    
    # 创建新的训练器并加载模型
    trainer2 = CBOWTrainer()
    trainer2.load_model(save_path)
    
    # 验证加载的模型
    assert trainer2.vocab_size == trainer1.vocab_size, "词汇表大小不匹配"
    assert trainer2.embedding_dim == trainer1.embedding_dim, "嵌入维度不匹配"
    
    if 'the' in trainer2.word_to_idx and original_embedding is not None:
        loaded_embedding = trainer2.get_word_embedding('the')
        if loaded_embedding is not None:
            assert np.allclose(original_embedding, loaded_embedding), "词嵌入不匹配"
    
    print("✓ 模型保存和加载功能正常")
    
    # 清理测试文件
    import os
    if os.path.exists(save_path):
        os.remove(save_path)


def test_edge_cases():
    """测试边缘情况"""
    print("\n=== 测试边缘情况 ===")
    
    trainer = CBOWTrainer(embedding_dim=10, context_size=1, min_count=1)
    
    # 测试空语料库
    try:
        trainer.train([], epochs=1)
        print("✓ 空语料库处理正常")
    except Exception as e:
        print(f"✓ 空语料库抛出异常: {type(e).__name__}")
    
    # 测试单句语料库
    single_sentence = ["hello world"]
    try:
        trainer.train(single_sentence, epochs=1)
        print("✓ 单句语料库处理正常")
    except Exception as e:
        print(f"! 单句语料库处理异常: {e}")
    
    # 测试不存在的词汇
    trainer.build_vocabulary(["hello world test"])
    embedding = trainer.get_word_embedding("nonexistent")
    assert embedding is None, "不存在词汇应返回None"
    
    similar = trainer.find_similar_words("nonexistent")
    assert similar == [], "不存在词汇的相似词应返回空列表"
    
    print("✓ 边缘情况处理正常")


def run_all_tests():
    """运行所有测试"""
    print("开始CBOW模型测试...\n")
    
    try:
        test_cbow_model_architecture()
        test_trainer_functionality()
        test_model_save_load()
        test_edge_cases()
        
        print("\n" + "="*50)
        print("✅ 所有测试通过!")
        print("CBOW模型实现验证成功")
        print("="*50)
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
