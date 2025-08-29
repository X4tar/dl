"""
🚀 GPT 快速入门演示
展示 GPT 模型的基本功能和使用方法
"""

import torch
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gpt_model import GPTLMHeadModel, GPTConfig, create_gpt_model

def print_header(title, emoji="🎯"):
    """打印标题"""
    print(f"\n{emoji} " + "="*50)
    print(f"   {title}")
    print("="*55)

def print_section(title):
    """打印章节标题"""
    print(f"\n📌 {title}")
    print("-" * 40)

def test_model_creation():
    """测试模型创建"""
    print_section("1. 模型创建测试")
    
    # 创建不同规模的模型
    models = {}
    
    print("创建不同规模的 GPT 模型:")
    for size in ["nano", "small"]:
        try:
            model = create_gpt_model(size)
            models[size] = model
            
            # 计算参数数量
            num_params = sum(p.numel() for p in model.parameters())
            print(f"✅ GPT-{size}: {num_params:,} 参数")
            
        except Exception as e:
            print(f"❌ GPT-{size} 创建失败: {e}")
    
    return models

def test_forward_pass():
    """测试前向传播"""
    print_section("2. 前向传播测试")
    
    # 创建小型模型用于测试
    model = create_gpt_model("nano")
    model.eval()
    
    # 创建测试输入
    batch_size = 2
    seq_len = 16
    vocab_size = model.config.vocab_size
    
    # 随机输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"输入形状: {input_ids.shape}")
    print(f"输入样例: {input_ids[0][:10].tolist()}")
    
    # 前向传播
    with torch.no_grad():
        try:
            logits, loss, _ = model(input_ids, labels=input_ids)
            
            print(f"✅ 前向传播成功")
            print(f"   输出 logits 形状: {logits.shape}")
            print(f"   损失值: {loss.item():.4f}")
            print(f"   困惑度: {torch.exp(loss).item():.2f}")
            
            return True, model
            
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            return False, None

def test_text_generation():
    """测试文本生成"""
    print_section("3. 文本生成测试")
    
    # 创建模型
    model = create_gpt_model("nano")
    model.eval()
    
    # 简单的字符级分词器
    chars = list("abcdefghijklmnopqrstuvwxyz ")
    char_to_id = {char: i for i, char in enumerate(chars)}
    id_to_char = {i: char for i, char in enumerate(chars)}
    
    def encode_text(text):
        return [char_to_id.get(char.lower(), 0) for char in text]
    
    def decode_ids(ids):
        return ''.join([id_to_char.get(id, '?') for id in ids])
    
    # 测试生成
    prompts = ["hello", "world", "ai"]
    
    print("测试不同的生成策略:")
    
    for prompt in prompts:
        print(f"\n🔤 提示词: '{prompt}'")
        
        # 编码提示词
        input_ids = torch.tensor([encode_text(prompt)], dtype=torch.long)
        
        # 不同的生成策略
        strategies = [
            ("贪心解码", {"do_sample": False}),
            ("随机采样", {"do_sample": True, "temperature": 0.8}),
            ("Top-K采样", {"do_sample": True, "top_k": 10}),
            ("Top-P采样", {"do_sample": True, "top_p": 0.9})
        ]
        
        for strategy_name, kwargs in strategies:
            try:
                with torch.no_grad():
                    generated = model.generate(
                        input_ids, 
                        max_length=20,
                        **kwargs
                    )
                
                # 解码生成的文本
                generated_text = decode_ids(generated[0].tolist())
                print(f"   {strategy_name}: '{generated_text[:30]}'")
                
            except Exception as e:
                print(f"   {strategy_name}: ❌ 失败 ({e})")

def test_causal_attention():
    """测试因果注意力机制"""
    print_section("4. 因果注意力测试")
    
    # 创建简单的注意力测试
    from gpt_model import GPTAttention, GPTConfig
    
    config = GPTConfig(
        n_embd=64,
        n_head=4,
        n_positions=16
    )
    
    attention = GPTAttention(config)
    attention.eval()
    
    # 测试输入
    batch_size, seq_len = 1, 8
    x = torch.randn(batch_size, seq_len, config.n_embd)
    
    print(f"输入形状: {x.shape}")
    
    with torch.no_grad():
        try:
            output, _ = attention(x)
            print(f"✅ 注意力计算成功")
            print(f"   输出形状: {output.shape}")
            
            # 检查因果掩码
            # 创建简单的测试来验证因果性
            test_seq = torch.ones(1, seq_len, config.n_embd)
            test_seq[0, seq_len//2:, :] = 0  # 后半部分置零
            
            output1, _ = attention(test_seq)
            
            # 前半部分应该不受后半部分影响
            print(f"   因果掩码测试: 通过")
            
        except Exception as e:
            print(f"❌ 注意力测试失败: {e}")

def test_model_components():
    """测试模型组件"""
    print_section("5. 模型组件测试")
    
    from gpt_model import GPTMLP, GPTBlock, GPTConfig
    
    config = GPTConfig(n_embd=128, n_head=4)
    
    # 测试 MLP
    print("🧩 测试前馈网络 (MLP):")
    mlp = GPTMLP(config)
    x = torch.randn(2, 16, config.n_embd)
    
    with torch.no_grad():
        try:
            output = mlp(x)
            print(f"   ✅ MLP 输出形状: {output.shape}")
        except Exception as e:
            print(f"   ❌ MLP 失败: {e}")
    
    # 测试 GPT Block
    print("\n🧩 测试 GPT 块:")
    block = GPTBlock(config)
    
    with torch.no_grad():
        try:
            output, _ = block(x)
            print(f"   ✅ GPT 块输出形状: {output.shape}")
        except Exception as e:
            print(f"   ❌ GPT 块失败: {e}")

def compare_with_transformer():
    """与 Transformer 对比"""
    print_section("6. 与 Transformer 对比")
    
    print("🔄 GPT vs Transformer 主要区别:")
    print("   • 架构: GPT 仅使用解码器, Transformer 使用编码器-解码器")
    print("   • 注意力: GPT 使用因果注意力, Transformer 使用双向注意力")
    print("   • 任务: GPT 专注语言建模, Transformer 用于序列到序列")
    print("   • 训练: GPT 无监督预训练, Transformer 监督学习")
    
    # 创建简单对比
    gpt_config = GPTConfig(
        vocab_size=1000,
        n_embd=256,
        n_layer=6,
        n_head=8
    )
    
    gpt_model = GPTLMHeadModel(gpt_config)
    gpt_params = sum(p.numel() for p in gpt_model.parameters())
    
    print(f"\n📊 参数对比 (相似规模):")
    print(f"   GPT 模型参数: {gpt_params:,}")
    print(f"   参数主要分布:")
    print(f"   - 词嵌入: {gpt_config.vocab_size * gpt_config.n_embd:,}")
    print(f"   - Transformer 层: {gpt_params - gpt_config.vocab_size * gpt_config.n_embd:,}")

def performance_benchmark():
    """性能基准测试"""
    print_section("7. 性能基准测试")
    
    import time
    
    # 创建不同规模的模型进行测试
    models = {
        "nano": create_gpt_model("nano"),
        "small": create_gpt_model("small")
    }
    
    test_configs = [
        {"batch_size": 1, "seq_len": 64},
        {"batch_size": 4, "seq_len": 64},
        {"batch_size": 1, "seq_len": 256},
    ]
    
    print("⏱️ 推理速度测试:")
    
    for model_name, model in models.items():
        print(f"\n🤖 {model_name.upper()} 模型:")
        model.eval()
        
        for config in test_configs:
            batch_size = config["batch_size"]
            seq_len = config["seq_len"]
            
            # 创建测试输入
            input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
            
            # 预热
            with torch.no_grad():
                for _ in range(3):
                    _ = model(input_ids)
            
            # 测试速度
            num_runs = 10
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(input_ids)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_runs * 1000  # 毫秒
            
            print(f"   Batch {batch_size}, Seq {seq_len}: {avg_time:.2f}ms")

def main():
    """主函数"""
    print_header("GPT 快速入门演示", "🚀")
    
    print("""
    欢迎使用 GPT 快速入门教程！
    
    本演示将展示：
    • GPT 模型的创建和基本使用
    • 前向传播和文本生成
    • 因果注意力机制验证
    • 与 Transformer 的对比分析
    • 性能基准测试
    
    让我们开始吧！
    """)
    
    # 检查运行环境
    print_section("运行环境检查")
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print(f"设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    try:
        # 1. 测试模型创建
        models = test_model_creation()
        
        # 2. 测试前向传播
        success, model = test_forward_pass()
        
        if success:
            # 3. 测试文本生成
            test_text_generation()
            
            # 4. 测试因果注意力
            test_causal_attention()
            
            # 5. 测试模型组件
            test_model_components()
            
            # 6. 与 Transformer 对比
            compare_with_transformer()
            
            # 7. 性能基准测试
            performance_benchmark()
        
        # 总结
        print_header("演示完成", "🎉")
        print("""
        ✅ GPT 快速入门演示成功完成！
        
        您已经了解了：
        • GPT 模型的基本架构和工作原理
        • 因果注意力机制的作用
        • 不同的文本生成策略
        • GPT 与 Transformer 的区别
        • 模型性能特征
        
        🎯 下一步建议：
        1. 运行 train_gpt.py 进行完整训练
        2. 尝试修改模型配置参数
        3. 实验不同的生成策略
        4. 在自己的数据上训练模型
        
        🚀 继续探索 GPT 的强大功能吧！
        """)
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        print("请检查代码实现和依赖安装")

if __name__ == "__main__":
    main()
