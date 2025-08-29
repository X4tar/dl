"""
Transformer 示例运行脚本
提供一个简单的命令行界面来运行各种示例
"""

import sys
import os
import torch

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_menu():
    """打印菜单"""
    print("\nTransformer 学习示例菜单:")
    print("-" * 40)
    print("1. 测试基础组件")
    print("2. 分析位置编码")  
    print("3. 测试完整模型")
    print("4. 训练翻译模型")
    print("5. 注意力可视化")
    print("6. 文本生成示例")
    print("7. 运行所有示例")
    print("0. 退出")
    print("-" * 40)

def test_components():
    """测试基础组件"""
    print_header("测试 Transformer 基础组件")
    try:
        import transformer_components
        
        # 直接运行组件测试
        print("运行组件测试...")
        if hasattr(transformer_components, 'test_all_components'):
            transformer_components.test_all_components()
        else:
            print("正在导入和测试组件...")
            # 创建简单测试
            from transformer_components import MultiHeadAttention, PositionwiseFeedForward
            
            batch_size, seq_len, d_model = 2, 10, 64
            x = torch.randn(batch_size, seq_len, d_model)
            
            # 测试多头注意力
            attn = MultiHeadAttention(d_model, n_heads=4)
            output, weights = attn(x, x, x)
            print(f"✓ 多头注意力测试通过，输出形状: {output.shape}")
            
            # 测试前馈网络
            ff = PositionwiseFeedForward(d_model, d_ff=256)
            output = ff(x)
            print(f"✓ 前馈网络测试通过，输出形状: {output.shape}")
        
    except Exception as e:
        print(f"运行组件测试时出错: {e}")
        print("请确保所有依赖都已正确安装")

def analyze_positional_encoding():
    """分析位置编码"""
    print_header("位置编码分析")
    try:
        from positional_encoding import PositionalEncoding
        
        print("运行位置编码分析...")
        # 创建位置编码测试
        d_model = 64
        max_seq_len = 100
        pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # 测试位置编码
        batch_size, seq_len = 2, 20
        x = torch.randn(batch_size, seq_len, d_model)
        x_with_pos = pos_encoding(x)
        
        print(f"✓ 位置编码测试通过")
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {x_with_pos.shape}")
        print(f"  位置编码是否改变输入: {not torch.equal(x, x_with_pos)}")
        
    except Exception as e:
        print(f"运行位置编码分析时出错: {e}")
        print("如果缺少 matplotlib，可能无法生成可视化图表")

def test_transformer_model():
    """测试完整的 Transformer 模型"""
    print_header("测试完整 Transformer 模型")
    try:
        from transformer_model import Transformer, TransformerForLanguageModeling
        
        print("运行模型测试...")
        
        # 测试编码器-解码器模型
        print("测试编码器-解码器 Transformer...")
        model = Transformer(
            src_vocab_size=100, tgt_vocab_size=100,
            d_model=128, n_heads=4, n_layers=2, d_ff=256
        )
        
        batch_size = 2
        src = torch.randint(0, 100, (batch_size, 10))
        tgt = torch.randint(0, 100, (batch_size, 8))
        
        model.eval()
        with torch.no_grad():
            output, _, _, _ = model(src, tgt)
        
        print(f"✓ 编码器-解码器模型测试通过，输出形状: {output.shape}")
        
        # 测试语言模型
        print("测试语言建模 Transformer...")
        lm_model = TransformerForLanguageModeling(
            vocab_size=100, d_model=128, n_heads=4, n_layers=2, d_ff=256
        )
        
        input_ids = torch.randint(0, 100, (batch_size, 15))
        
        lm_model.eval()
        with torch.no_grad():
            logits, attention_weights = lm_model(input_ids)
        
        print(f"✓ 语言模型测试通过，输出形状: {logits.shape}")
        print(f"✓ 注意力层数: {len(attention_weights)}")
        
    except Exception as e:
        print(f"运行模型测试时出错: {e}")
        import traceback
        traceback.print_exc()

def train_translation_model():
    """训练翻译模型"""
    print_header("训练翻译模型")
    
    # 警告用户训练时间
    print("警告: 训练过程可能需要几分钟时间")
    print("这将训练一个简单的英法翻译模型")
    
    user_input = input("是否继续? (y/n): ").lower()
    if user_input != 'y':
        print("已取消训练")
        return
    
    try:
        from train_transformer import main
        main()
        
    except Exception as e:
        print(f"运行训练时出错: {e}")
        print("这可能是因为缺少依赖或设备资源不足")

def visualize_attention():
    """注意力可视化"""
    print_header("注意力机制可视化")
    
    print("注意: 可视化需要 matplotlib 和 seaborn")
    user_input = input("是否继续? (y/n): ").lower()
    if user_input != 'y':
        print("已取消可视化")
        return
    
    try:
        exec(open('transfomer/attention_visualization.py').read())
        
    except Exception as e:
        print(f"运行注意力可视化时出错: {e}")
        print("请确保安装了 matplotlib 和 seaborn: pip install matplotlib seaborn")

def text_generation_example():
    """文本生成示例"""
    print_header("文本生成示例")
    
    print("警告: 文本生成训练可能需要几分钟时间")
    print("这将训练一个简单的语言模型并生成莎士比亚风格的文本")
    
    user_input = input("是否继续? (y/n): ").lower()
    if user_input != 'y':
        print("已取消文本生成")
        return
    
    try:
        exec(open('transfomer/text_generation_example.py').read())
        
    except Exception as e:
        print(f"运行文本生成时出错: {e}")

def run_all_examples():
    """运行所有示例"""
    print_header("运行所有示例")
    
    print("警告: 运行所有示例可能需要较长时间")
    print("建议单独运行各个示例以便更好地理解")
    
    user_input = input("是否继续运行所有示例? (y/n): ").lower()
    if user_input != 'y':
        print("已取消")
        return
    
    examples = [
        ("基础组件测试", test_components),
        ("位置编码分析", analyze_positional_encoding),
        ("模型测试", test_transformer_model),
        ("训练翻译模型", train_translation_model),
        ("注意力可视化", visualize_attention),
        ("文本生成示例", text_generation_example)
    ]
    
    for name, func in examples:
        print(f"\n开始运行: {name}")
        try:
            func()
            print(f"✓ {name} 完成")
        except Exception as e:
            print(f"✗ {name} 失败: {e}")
        
        input("\n按 Enter 继续下一个示例...")

def check_dependencies():
    """检查依赖"""
    print_header("检查依赖")
    
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib (可选，用于可视化)',
        'seaborn': 'Seaborn (可选，用于可视化)'
    }
    
    print("检查必要的 Python 包:")
    print("-" * 30)
    
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {package} - {description}")
        except ImportError:
            print(f"✗ {package} - {description} (未安装)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少的包: {', '.join(missing_packages)}")
        print("安装命令:")
        for package in missing_packages:
            if package in ['matplotlib', 'seaborn']:
                print(f"  pip install {package}  # 可选")
            else:
                print(f"  pip install {package}  # 必需")
    else:
        print("\n✓ 所有依赖都已安装!")
    
    # 检查 CUDA
    print(f"\nPyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {'是' if torch.cuda.is_available() else '否'}")
    if torch.cuda.is_available():
        print(f"CUDA 设备数量: {torch.cuda.device_count()}")
        print(f"当前设备: {torch.cuda.get_device_name(0)}")

def show_project_structure():
    """显示项目结构"""
    print_header("项目结构")
    
    print("transfomer/ 目录结构:")
    print("├── README_transformer_tutorial.md  # 完整教程文档")
    print("├── transformer_components.py       # 基础组件实现")
    print("├── positional_encoding.py          # 位置编码详解")
    print("├── transformer_model.py            # 完整模型实现")
    print("├── train_transformer.py            # 训练脚本")
    print("├── attention_visualization.py      # 注意力可视化")
    print("├── text_generation_example.py      # 文本生成示例")
    print("└── run_examples.py                 # 示例运行脚本")
    
    print("\n学习建议顺序:")
    print("1. 阅读 README_transformer_tutorial.md")
    print("2. 运行基础组件测试，理解各个组件")
    print("3. 分析位置编码，理解位置信息的重要性")
    print("4. 测试完整模型，了解整体架构")
    print("5. 训练翻译模型，体验实际应用")
    print("6. 可视化注意力，深入理解注意力机制")
    print("7. 文本生成示例，探索语言建模")

def main():
    """主函数"""
    print_header("Transformer 完整学习示例")
    print("欢迎使用 Transformer 学习工具!")
    print("本工具包含完整的 Transformer 实现和教学示例")
    
    # 首先检查依赖
    check_dependencies()
    
    while True:
        print_menu()
        
        try:
            choice = input("\n请选择 (0-7): ").strip()
            
            if choice == '0':
                print("感谢使用 Transformer 学习工具!")
                break
            elif choice == '1':
                test_components()
            elif choice == '2':
                analyze_positional_encoding()
            elif choice == '3':
                test_transformer_model()
            elif choice == '4':
                train_translation_model()
            elif choice == '5':
                visualize_attention()
            elif choice == '6':
                text_generation_example()
            elif choice == '7':
                run_all_examples()
            elif choice == '8':
                show_project_structure()
            elif choice == 'h' or choice == 'help':
                show_project_structure()
            else:
                print("无效选择，请输入 0-7")
                
        except KeyboardInterrupt:
            print("\n\n用户中断，退出程序")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            
        input("\n按 Enter 返回主菜单...")

if __name__ == "__main__":
    main()
