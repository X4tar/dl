"""
Transformer 快速入门脚本
用于测试基本功能的简单脚本
"""

import torch
import torch.nn as nn
import numpy as np

def test_basic_components():
    """测试基础组件"""
    print("=" * 50)
    print("测试 Transformer 基础组件")
    print("=" * 50)
    
    # 测试基础组件是否可以导入和运行
    try:
        from transformer_components import MultiHeadAttention, PositionwiseFeedForward
        
        # 创建测试数据
        batch_size = 2
        seq_len = 10
        d_model = 64
        
        x = torch.randn(batch_size, seq_len, d_model)
        print(f"输入形状: {x.shape}")
        
        # 测试多头注意力
        multi_head_attn = MultiHeadAttention(d_model, n_heads=4)
        attn_output, attn_weights = multi_head_attn(x, x, x)
        print(f"多头注意力输出形状: {attn_output.shape}")
        print(f"注意力权重形状: {attn_weights.shape}")
        
        # 测试前馈网络
        ff = PositionwiseFeedForward(d_model, d_ff=256)
        ff_output = ff(x)
        print(f"前馈网络输出形状: {ff_output.shape}")
        
        print("✓ 基础组件测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 基础组件测试失败: {e}")
        return False

def test_positional_encoding():
    """测试位置编码"""
    print("\n" + "=" * 50)
    print("测试位置编码")
    print("=" * 50)
    
    try:
        from positional_encoding import PositionalEncoding
        
        # 测试位置编码
        d_model = 64
        seq_len = 20
        batch_size = 2
        
        pos_encoding = PositionalEncoding(d_model, max_seq_len=100)
        x = torch.randn(batch_size, seq_len, d_model)
        
        x_with_pos = pos_encoding(x)
        print(f"原始输入形状: {x.shape}")
        print(f"添加位置编码后形状: {x_with_pos.shape}")
        
        # 验证位置编码的作用
        print(f"位置编码前后是否相同: {torch.equal(x, x_with_pos)}")
        
        print("✓ 位置编码测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 位置编码测试失败: {e}")
        return False

def test_transformer_model():
    """测试完整模型"""
    print("\n" + "=" * 50)
    print("测试完整 Transformer 模型")
    print("=" * 50)
    
    try:
        from transformer_model import Transformer, TransformerForLanguageModeling
        
        # 测试编码器-解码器模型
        src_vocab_size = 100
        tgt_vocab_size = 100
        model = Transformer(
            src_vocab_size, tgt_vocab_size,
            d_model=128, n_heads=4, n_layers=2, d_ff=256
        )
        
        batch_size = 2
        src_seq_len = 10
        tgt_seq_len = 8
        
        src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
        tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
        
        output, _, _, _ = model(src, tgt)
        print(f"编码器-解码器模型输出形状: {output.shape}")
        
        # 测试语言模型
        lm_model = TransformerForLanguageModeling(
            vocab_size=100, d_model=128, n_heads=4, n_layers=2, d_ff=256
        )
        
        input_ids = torch.randint(0, 100, (batch_size, 15))
        logits, attention_weights = lm_model(input_ids)
        print(f"语言模型输出形状: {logits.shape}")
        print(f"注意力层数: {len(attention_weights)}")
        
        print("✓ 模型测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        return False

def demo_simple_training():
    """演示简单训练"""
    print("\n" + "=" * 50)
    print("演示简单训练过程")
    print("=" * 50)
    
    try:
        from transformer_model import TransformerForLanguageModeling
        
        # 创建小模型
        model = TransformerForLanguageModeling(
            vocab_size=50, d_model=64, n_heads=4, n_layers=2, d_ff=128
        )
        
        # 创建随机数据
        batch_size = 4
        seq_len = 10
        data = torch.randint(0, 50, (batch_size, seq_len))
        targets = torch.randint(0, 50, (batch_size, seq_len))
        
        # 设置优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print("开始简单训练演示...")
        
        # 训练几步
        for step in range(5):
            optimizer.zero_grad()
            
            logits, _ = model(data)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            print(f"步骤 {step+1}, 损失: {loss.item():.4f}")
        
        print("✓ 简单训练演示完成")
        return True
        
    except Exception as e:
        print(f"✗ 训练演示失败: {e}")
        return False

def demo_text_generation():
    """演示文本生成"""
    print("\n" + "=" * 50)
    print("演示文本生成")
    print("=" * 50)
    
    try:
        from transformer_model import TransformerForLanguageModeling
        
        # 创建模型
        vocab_size = 20
        model = TransformerForLanguageModeling(
            vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=2, d_ff=128
        )
        
        # 简单的生成示例
        model.eval()
        with torch.no_grad():
            # 从随机输入开始
            input_ids = torch.randint(0, vocab_size, (1, 5))
            print(f"初始输入: {input_ids[0].tolist()}")
            
            # 生成几个token
            generated = input_ids.clone()
            for i in range(10):
                logits, _ = model(generated)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
            
            print(f"生成序列: {generated[0].tolist()}")
        
        print("✓ 文本生成演示完成")
        return True
        
    except Exception as e:
        print(f"✗ 文本生成演示失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("开始运行 Transformer 快速测试套件...")
    
    tests = [
        ("基础组件", test_basic_components),
        ("位置编码", test_positional_encoding),
        ("完整模型", test_transformer_model),
        ("简单训练", demo_simple_training),
        ("文本生成", demo_text_generation)
    ]
    
    results = []
    
    for name, test_func in tests:
        success = test_func()
        results.append((name, success))
    
    # 打印总结
    print("\n" + "=" * 50)
    print("测试结果总结")
    print("=" * 50)
    
    passed = 0
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{name}: {status}")
        if success:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 个测试通过")
    
    if passed == len(results):
        print("\n🎉 所有测试都通过了！您的 Transformer 实现工作正常。")
        print("\n接下来您可以:")
        print("1. 运行 train_transformer.py 进行完整训练")
        print("2. 运行 attention_visualization.py 查看注意力可视化")
        print("3. 运行 text_generation_example.py 体验文本生成")
    else:
        print(f"\n⚠️  有 {len(results) - passed} 个测试失败")
        print("请检查依赖是否正确安装：")
        print("pip install torch numpy")

def check_environment():
    """检查环境"""
    print("检查运行环境...")
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"NumPy 版本: {np.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    
    print()

if __name__ == "__main__":
    import sys
    
    print("🚀 Transformer 快速入门测试")
    print("这个脚本将测试 Transformer 实现的基本功能")
    print()
    
    # 检查环境
    check_environment()
    
    # 运行测试
    run_all_tests()
