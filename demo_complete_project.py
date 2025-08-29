#!/usr/bin/env python3
"""
🎓 完整 Transformer 教学项目演示脚本
展示项目中所有模块的核心功能
"""

import os
import sys
import time
import subprocess

def print_header(title, emoji="🎯"):
    """打印标题"""
    print(f"\n{emoji} " + "="*60)
    print(f"   {title}")
    print("="*65)

def print_subheader(title):
    """打印子标题"""
    print(f"\n📌 {title}")
    print("-" * 50)

def run_module_demo(module_name, script_name, timeout=30):
    """运行模块演示"""
    print(f"🚀 正在运行 {module_name} 演示...")
    
    try:
        # 使用 subprocess 运行脚本，设置超时
        result = subprocess.run(
            ["uv", "run", "python", script_name],
            cwd=module_name,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            print(f"✅ {module_name} 演示运行成功！")
            # 显示部分输出
            output_lines = result.stdout.split('\n')
            for i, line in enumerate(output_lines):
                if i < 10:  # 显示前10行
                    print(f"   {line}")
                elif i == 10:
                    print("   ...")
                    break
            return True
        else:
            print(f"❌ {module_name} 演示运行失败")
            print(f"错误: {result.stderr[:200]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {module_name} 演示超时（{timeout}秒）")
        return False
    except Exception as e:
        print(f"❌ 运行 {module_name} 时出错: {e}")
        return False

def show_project_structure():
    """显示项目结构"""
    print_header("项目结构概览", "🗂️")
    
    structure = """
📁 完整 Transformer 教学项目
├── 📁 transfomer/           # 🔄 原始 Transformer 实现
│   ├── 📖 README_transformer_tutorial.md
│   ├── 🧩 transformer_components.py
│   ├── 📐 positional_encoding.py  
│   ├── 🤖 transformer_model.py
│   ├── 🎯 train_transformer.py
│   ├── 👁️ attention_visualization.py
│   ├── ✍️ text_generation_example.py
│   └── 🚀 quick_start.py
│
├── 📁 vit/                  # 🖼️ Vision Transformer 实现
│   ├── 📖 README_vit_tutorial.md
│   ├── 🧩 vit_components.py
│   ├── 🔍 patch_embedding.py
│   ├── 🤖 vit_model.py
│   ├── 🎯 vit_trainer.py
│   └── 🚀 quick_start.py
│
├── 📁 bert/                 # 🤖 BERT 实现
│   ├── 📖 README_bert_tutorial.md
│   ├── 🧩 bert_components.py
│   ├── 🤖 bert_model.py
│   ├── 🎓 bert_pretraining.py
│   └── 🚀 quick_start.py
│
└── 📁 cbow/                 # 📝 传统词嵌入对比
    ├── cbow_example.py
    └── cbow_model.py
    """
    print(structure)

def show_learning_path():
    """显示学习路径"""
    print_header("建议学习路径", "🎓")
    
    print("""
🌟 第一阶段：理论基础 (1-2天)
   📖 阅读教程文档，理解核心概念
   🔍 了解注意力机制、位置编码等基本原理
   
🛠️ 第二阶段：代码实践 (3-5天)  
   🧩 分析基础组件实现
   🚀 运行快速入门脚本
   👁️ 观察注意力可视化
   
🎯 第三阶段：模型训练 (1-2周)
   🎲 尝试不同的模型配置
   📊 分析训练过程和结果
   🔬 进行消融实验
   
🚀 第四阶段：应用扩展 (持续)
   💡 实现自己的模型变体
   🌐 应用到实际项目中
   📝 贡献改进和优化
    """)

def demonstrate_key_concepts():
    """演示关键概念"""
    print_header("核心概念演示", "💡")
    
    print_subheader("1. 注意力机制")
    print("""
    🧠 注意力机制让模型能够：
    • 关注输入序列的不同部分
    • 计算序列中每个位置的重要性
    • 捕获长距离依赖关系
    
    📊 数学原理：
    Attention(Q,K,V) = softmax(QK^T/√d_k)V
    """)
    
    print_subheader("2. Transformer 架构")
    print("""
    🔄 Transformer 的创新：
    • 完全基于注意力机制，摒弃递归和卷积
    • 编码器-解码器结构
    • 多头注意力并行处理
    • 位置编码解决序列顺序问题
    """)
    
    print_subheader("3. 模型变体")
    print("""
    🤖 主要变体：
    • BERT: 双向编码器，擅长理解任务
    • GPT: 单向解码器，擅长生成任务  
    • ViT: 将图像视为序列，统一视觉和语言处理
    • T5: 文本到文本的统一框架
    """)

def run_all_demonstrations():
    """运行所有演示"""
    print_header("🎉 开始完整项目演示", "🎪")
    
    results = {}
    
    print_subheader("测试项目模块")
    
    # 演示各个模块
    modules = [
        ("transfomer", "quick_start.py", "原始 Transformer"),
        ("vit", "quick_start.py", "Vision Transformer"), 
        ("bert", "quick_start.py", "BERT 模型"),
        ("cbow", "cbow_example.py", "传统词嵌入对比")
    ]
    
    for module_dir, script, description in modules:
        if os.path.exists(module_dir):
            print(f"\n🔍 演示 {description}...")
            success = run_module_demo(module_dir, script, timeout=60)
            results[description] = success
            time.sleep(1)  # 短暂停顿
        else:
            print(f"⚠️ 模块 {module_dir} 不存在")
            results[description] = False
    
    return results

def show_results_summary(results):
    """显示结果总结"""
    print_header("演示结果总结", "📊")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"📈 总共测试了 {total} 个模块")
    print(f"✅ 成功运行: {passed} 个")
    print(f"❌ 运行失败: {total - passed} 个")
    print(f"🎯 成功率: {passed/total*100:.1f}%")
    
    print("\n详细结果:")
    for module, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {module}")
    
    if passed == total:
        print("\n🎉 恭喜！所有模块都运行成功！")
        print("🚀 您已经拥有了一个完整可用的 Transformer 教学项目！")
    else:
        print(f"\n⚠️ 有 {total - passed} 个模块需要检查")
        print("💡 建议检查依赖安装和代码实现")

def show_next_steps():
    """显示后续步骤"""
    print_header("接下来可以做什么？", "🚀")
    
    print("""
🎓 深入学习：
  • 📖 仔细阅读各模块的 README 文档
  • 🔍 分析代码实现细节
  • 🧪 尝试修改参数进行实验
  
🛠️ 实践项目：
  • 🎯 在自己的数据上训练模型
  • 📊 可视化不同层的注意力权重
  • 🔬 进行消融实验分析各组件作用
  
🚀 进阶挑战：
  • 💡 实现新的模型变体
  • 🌐 集成到实际应用中
  • 📝 优化模型性能和效率
  
🤝 社区贡献：
  • 🐛 报告和修复 bug
  • 📚 完善文档和注释
  • 💡 分享使用经验和改进建议
    """)

def main():
    """主函数"""
    print_header("🎓 完整 Transformer 教学项目演示", "🎪")
    
    print("""
    欢迎使用完整 Transformer 教学项目！ 
    
    这个项目包含：
    🔄 原始 Transformer (编码器-解码器)
    🖼️ Vision Transformer (图像分类)  
    🤖 BERT (双向编码器)
    📝 传统词嵌入对比 (CBOW)
    
    让我们开始演示吧！
    """)
    
    # 显示项目结构
    show_project_structure()
    
    # 显示学习路径
    show_learning_path()
    
    # 演示核心概念
    demonstrate_key_concepts()
    
    # 询问是否运行完整演示
    print("\n" + "="*60)
    response = input("🤔 是否运行完整的模块演示？这可能需要几分钟时间 (y/n): ")
    
    if response.lower() in ['y', 'yes', '是', '好的']:
        # 运行所有演示
        results = run_all_demonstrations()
        
        # 显示结果总结
        show_results_summary(results)
        
        # 显示后续步骤
        show_next_steps()
    else:
        print("\n✨ 演示已跳过。您可以随时运行各模块的 quick_start.py 来体验功能！")
        show_next_steps()
    
    print_header("感谢使用 Transformer 教学项目！", "🙏")
    print("""
    🌟 希望这个项目能帮助您：
    • 深入理解 Transformer 架构
    • 掌握现代深度学习技术
    • 在实际项目中应用所学知识
    
    🎓 学习愉快，编程愉快！
    
    📧 如有问题或建议，欢迎反馈！
    """)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，感谢使用！")
    except Exception as e:
        print(f"\n❌ 运行时出错: {e}")
        print("💡 请检查项目完整性和依赖安装")
