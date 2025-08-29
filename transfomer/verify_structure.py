"""
代码结构验证脚本
检查所有文件是否存在且结构正确，不依赖外部库
"""

import os
import sys

def check_file_exists(filepath):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"✓ {filepath}")
        return True
    else:
        print(f"✗ {filepath} - 文件不存在")
        return False

def check_python_syntax(filepath):
    """检查 Python 文件语法"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 编译检查语法
        compile(code, filepath, 'exec')
        return True
    except SyntaxError as e:
        print(f"  语法错误: {e}")
        return False
    except Exception as e:
        print(f"  检查错误: {e}")
        return False

def verify_project_structure():
    """验证项目结构"""
    print("=" * 60)
    print("Transformer 项目结构验证")
    print("=" * 60)
    
    # 必需文件列表
    required_files = [
        "transfomer/README_transformer_tutorial.md",
        "transfomer/transformer_components.py",
        "transfomer/positional_encoding.py",
        "transfomer/transformer_model.py",
        "transfomer/train_transformer.py",
        "transfomer/attention_visualization.py",
        "transfomer/text_generation_example.py",
        "transfomer/run_examples.py",
        "transfomer/quick_start.py"
    ]
    
    print("\n1. 检查文件存在性:")
    print("-" * 30)
    
    missing_files = []
    for filepath in required_files:
        if not check_file_exists(filepath):
            missing_files.append(filepath)
    
    if missing_files:
        print(f"\n缺少 {len(missing_files)} 个文件:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("\n2. 检查 Python 文件语法:")
    print("-" * 30)
    
    python_files = [f for f in required_files if f.endswith('.py')]
    syntax_errors = []
    
    for filepath in python_files:
        print(f"检查 {filepath}...")
        if not check_python_syntax(filepath):
            syntax_errors.append(filepath)
        else:
            print(f"  ✓ 语法正确")
    
    if syntax_errors:
        print(f"\n{len(syntax_errors)} 个文件有语法错误:")
        for file in syntax_errors:
            print(f"  - {file}")
        return False
    
    print("\n3. 检查文件大小:")
    print("-" * 30)
    
    for filepath in required_files:
        try:
            size = os.path.getsize(filepath)
            if size > 0:
                print(f"✓ {filepath}: {size} 字节")
            else:
                print(f"⚠ {filepath}: 空文件")
        except Exception as e:
            print(f"✗ {filepath}: 无法获取大小 - {e}")
    
    return True

def show_project_summary():
    """显示项目总结"""
    print("\n" + "=" * 60)
    print("项目总结")
    print("=" * 60)
    
    print("\n📚 Transformer 完整教学实现包含:")
    print("├── 📖 完整理论教程 (README)")
    print("├── 🧩 基础组件实现 (transformer_components.py)")
    print("├── 📍 位置编码详解 (positional_encoding.py)")
    print("├── 🏗️ 完整模型架构 (transformer_model.py)")
    print("├── 🚀 训练示例 (train_transformer.py)")
    print("├── 👁️ 注意力可视化 (attention_visualization.py)")
    print("├── ✍️ 文本生成示例 (text_generation_example.py)")
    print("├── 🎮 交互式运行器 (run_examples.py)")
    print("└── ⚡ 快速入门测试 (quick_start.py)")
    
    print("\n🎯 学习目标:")
    print("• 深入理解 Transformer 架构的每个组件")
    print("• 掌握注意力机制的数学原理和实现")
    print("• 学会从零实现完整的 Transformer 模型")
    print("• 体验真实的训练和推理过程")
    print("• 通过可视化直观理解模型工作原理")
    
    print("\n🚀 使用方法:")
    print("1. 安装依赖: pip install torch numpy matplotlib seaborn")
    print("2. 快速测试: python3 transfomer/quick_start.py")
    print("3. 逐步学习: 按 README 中的学习路径执行")
    print("4. 交互学习: python3 transfomer/run_examples.py")
    
    print("\n💡 特色功能:")
    print("• 详细的中文注释，适合中文学习者")
    print("• 渐进式学习设计，从简单到复杂")
    print("• 完整的训练样例，可直接运行")
    print("• 注意力可视化，直观理解机制")
    print("• 多种文本生成策略展示")

def main():
    """主函数"""
    print("🔍 Transformer 项目验证工具")
    print("这个脚本验证项目结构的完整性，不需要外部依赖")
    print()
    
    # 验证项目结构
    success = verify_project_structure()
    
    if success:
        print("\n🎉 项目结构验证通过！")
        print("所有必需文件都存在且语法正确。")
    else:
        print("\n❌ 项目结构验证失败！")
        print("请检查缺失的文件或语法错误。")
    
    # 显示项目总结
    show_project_summary()
    
    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
