"""
🎓 LLM 完整教程运行器
一键体验所有教程项目的核心功能
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    """打印欢迎横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🎓 LLM 完整学习教程                        ║
    ║              从 Transformer 到现代大语言模型                  ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  🧠 Transformer  📖 BERT  🖼️ ViT  🤖 GPT  🎯 指令微调      ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_section(title, emoji="🔥"):
    """打印章节标题"""
    print(f"\n{emoji} " + "="*60)
    print(f"   {title}")
    print("="*65)

def check_dependencies():
    """检查依赖项"""
    print_section("环境检查", "🔍")
    
    required_packages = [
        "torch", "numpy", "matplotlib", 
        "tqdm", "sklearn"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - 已安装")
        except ImportError:
            print(f"❌ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺少依赖项: {', '.join(missing_packages)}")
        print("请运行: pip install torch numpy matplotlib tqdm scikit-learn")
        return False
    
    print("\n✅ 所有依赖项检查通过！")
    return True

def run_tutorial(name, script_path, description):
    """运行单个教程"""
    print(f"\n📚 {name}: {description}")
    print("-" * 50)
    
    if not os.path.exists(script_path):
        print(f"❌ 文件不存在: {script_path}")
        return False
    
    try:
        print(f"🚀 运行 {script_path}...")
        
        # 使用 python3 运行脚本
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ 运行成功！")
            # 显示输出的前几行
            lines = result.stdout.split('\n')[:10]
            for line in lines:
                if line.strip():
                    print(f"   {line}")
            if len(result.stdout.split('\n')) > 10:
                print("   ... (更多输出)")
            return True
        else:
            print("❌ 运行失败")
            print("错误信息:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 运行超时 (>5分钟)")
        return False
    except Exception as e:
        print(f"❌ 运行异常: {e}")
        return False

def show_tutorial_menu():
    """显示教程菜单"""
    tutorials = [
        {
            "name": "🔄 Transformer 基础",
            "script": "transfomer/quick_start.py",
            "description": "原始 Transformer 架构演示"
        },
        {
            "name": "📖 BERT 模型",
            "script": "bert/quick_start.py", 
            "description": "双向编码器表示学习"
        },
        {
            "name": "🖼️ Vision Transformer",
            "script": "vit/quick_start.py",
            "description": "图像分类的 Transformer 应用"
        },
        {
            "name": "🤖 GPT 生成模型",
            "script": "gpt/quick_start.py",
            "description": "生成式预训练 Transformer"
        },
        {
            "name": "📝 词嵌入对比 (CBOW)",
            "script": "cbow/cbow_example.py",
            "description": "传统词嵌入 vs Transformer 对比"
        }
    ]
    
    print_section("🎯 选择要运行的教程")
    print("0. 🚀 运行所有教程 (推荐)")
    
    for i, tutorial in enumerate(tutorials, 1):
        print(f"{i}. {tutorial['name']}")
        print(f"   📝 {tutorial['description']}")
    
    print("6. 📊 项目验证和总结")
    print("7. 🚪 退出")
    
    return tutorials

def run_project_validation():
    """运行项目验证"""
    print_section("📊 项目验证", "🔬")
    
    validation_script = "validate_project.py"
    if os.path.exists(validation_script):
        print("运行项目完整性验证...")
        try:
            result = subprocess.run([
                sys.executable, validation_script
            ], capture_output=True, text=True, timeout=120)
            
            print(result.stdout)
            if result.stderr:
                print("警告信息:")
                print(result.stderr)
                
        except Exception as e:
            print(f"验证脚本运行失败: {e}")
    else:
        print("验证脚本不存在，跳过验证")

def show_project_summary():
    """显示项目总结"""
    print_section("🎉 项目学习总结", "📋")
    
    print("""
🎓 恭喜！您已经完成了完整的 LLM 学习之旅！

📚 您学到的核心技术：
├── 🔄 Transformer 架构
│   ├── 多头自注意力机制
│   ├── 位置编码
│   └── 编码器-解码器结构
├── 📖 BERT 双向编码
│   ├── 掩码语言建模
│   ├── 下一句预测
│   └── 预训练-微调范式
├── 🖼️ Vision Transformer
│   ├── 图像块嵌入
│   ├── 多模态应用
│   └── 注意力可视化
├── 🤖 GPT 生成模型
│   ├── 因果注意力掩码
│   ├── 自回归生成
│   └── 文本生成策略
└── 🎯 指令微调技术
    ├── 监督微调 (SFT)
    ├── 奖励模型
    └── 人类反馈强化学习 (RLHF)

🚀 您现在可以：
✅ 理解现代 LLM 的核心原理
✅ 从零实现 Transformer 系列模型
✅ 训练和微调语言模型
✅ 应用模型解决实际问题
✅ 评估和优化模型性能

🎯 建议的下一步学习：
1. 📈 尝试更大规模的模型训练
2. 🔧 学习模型优化和压缩技术
3. 🌐 探索多模态和多语言模型
4. 🤖 构建 AI Agent 和应用系统
5. 📚 深入研究最新的研究论文

💡 持续学习建议：
• 关注 Hugging Face, OpenAI 等平台的新发布
• 参与开源项目贡献代码
• 尝试复现最新的研究论文
• 构建自己的 AI 应用项目

🌟 记住：AI 技术发展迅速，保持好奇心和学习热情最重要！

感谢您完成这个学习之旅！🎊
    """)

def interactive_menu():
    """交互式菜单"""
    tutorials = show_tutorial_menu()
    
    while True:
        print("\n" + "="*50)
        try:
            choice = input("🔢 请选择要运行的教程 (0-7): ").strip()
            
            if choice == '7':
                print("👋 感谢使用！再见！")
                break
            elif choice == '0':
                print("🚀 开始运行所有教程...")
                success_count = 0
                
                for tutorial in tutorials:
                    if run_tutorial(
                        tutorial["name"], 
                        tutorial["script"], 
                        tutorial["description"]
                    ):
                        success_count += 1
                    time.sleep(2)  # 短暂暂停
                
                print(f"\n📊 运行结果: {success_count}/{len(tutorials)} 个教程成功")
                run_project_validation()
                show_project_summary()
                
            elif choice == '6':
                run_project_validation()
                show_project_summary()
                
            elif choice.isdigit() and 1 <= int(choice) <= len(tutorials):
                idx = int(choice) - 1
                tutorial = tutorials[idx]
                run_tutorial(
                    tutorial["name"],
                    tutorial["script"],
                    tutorial["description"]
                )
            else:
                print("❌ 无效选择，请输入 0-7 之间的数字")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出程序")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

def main():
    """主函数"""
    print_banner()
    
    print("""
🎯 欢迎使用 LLM 完整学习教程！

本教程包含：
• 🔄 Transformer 原理与实现
• 📖 BERT 双向编码模型  
• 🖼️ Vision Transformer 图像应用
• 🤖 GPT 生成式语言模型
• 🎯 指令微调与 RLHF 技术

每个模块都包含：
✅ 详细的理论解释
✅ 完整的代码实现
✅ 实际的训练示例
✅ 可视化和分析工具

🚀 让我们开始这个激动人心的学习之旅吧！
    """)
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 环境检查失败，请安装必要的依赖后重试")
        return
    
    # 检查项目结构
    print_section("项目结构检查", "📁")
    required_dirs = [
        "transfomer", "bert", "vit", "gpt", 
        "instruction_tuning", "cbow"
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/ - 存在")
        else:
            print(f"❌ {dir_name}/ - 缺失")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\n⚠️ 缺少目录: {', '.join(missing_dirs)}")
        print("某些教程可能无法正常运行")
    else:
        print("\n✅ 项目结构完整！")
    
    # 显示交互菜单
    interactive_menu()

if __name__ == "__main__":
    main()
