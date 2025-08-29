"""
ViT 项目结构验证脚本
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

def verify_vit_structure():
    """验证 ViT 项目结构"""
    print("=" * 60)
    print("Vision Transformer (ViT) 项目结构验证")
    print("=" * 60)
    
    # 必需文件列表
    required_files = [
        "vit/README_vit_tutorial.md",
        "vit/vit_components.py",
        "vit/patch_embedding.py",
        "vit/vit_model.py",
        "vit/vit_trainer.py",
        "vit/quick_start.py",
        "vit/run_examples.py",
        "vit/verify_vit_structure.py"
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

def show_vit_summary():
    """显示 ViT 项目总结"""
    print("\n" + "=" * 60)
    print("ViT 项目总结")
    print("=" * 60)
    
    print("\n🧠 Vision Transformer 完整教学实现包含:")
    print("├── 📖 完整理论教程 (README_vit_tutorial.md)")
    print("├── 🧩 基础组件实现 (vit_components.py)")
    print("├── 🖼️ 图像分块和嵌入 (patch_embedding.py)")
    print("├── 🏗️ 完整模型架构 (vit_model.py)")
    print("├── 🚀 训练和评估 (vit_trainer.py)")
    print("├── ⚡ 快速入门测试 (quick_start.py)")
    print("├── 🎮 交互式运行器 (run_examples.py)")
    print("└── 🔍 项目验证工具 (verify_vit_structure.py)")
    
    print("\n🎯 学习目标:")
    print("• 深入理解 Vision Transformer 架构")
    print("• 掌握图像分块和位置编码机制")
    print("• 学会从零实现完整的 ViT 模型")
    print("• 体验真实的计算机视觉训练流程")
    print("• 通过注意力可视化理解模型行为")
    
    print("\n🚀 使用方法:")
    print("1. 安装依赖: pip install torch torchvision numpy matplotlib pillow")
    print("2. 快速测试: python3 vit/quick_start.py")
    print("3. 逐步学习: 按 README 中的学习路径执行")
    print("4. 交互学习: python3 vit/run_examples.py")
    
    print("\n💡 核心特色:")
    print("• 完全从零实现的 ViT 架构")
    print("• 详细的中文注释和教学设计")
    print("• 图像分块机制的直观演示")
    print("• 多种模型配置 (Tiny/Small/Base/Large/Huge)")
    print("• 注意力机制的可视化分析")
    print("• ViT vs CNN 的深入对比")
    
    print("\n🔬 技术亮点:")
    print("• 标准 Transformer 编码器实现")
    print("• 高效的卷积-based patch embedding")
    print("• 可学习的位置编码")
    print("• CLS token 分类机制")
    print("• 完整的训练和推理流程")
    print("• 多头注意力权重分析")

def check_imports():
    """检查关键导入语句"""
    print("\n4. 检查关键导入:")
    print("-" * 30)
    
    import_checks = [
        ("torch", "PyTorch 深度学习框架"),
        ("numpy", "NumPy 数值计算库"),
        ("matplotlib", "Matplotlib 绘图库"),
        ("PIL", "Pillow 图像处理库"),
    ]
    
    missing_deps = []
    
    for module, description in import_checks:
        try:
            __import__(module)
            print(f"✓ {module} - {description}")
        except ImportError:
            print(f"✗ {module} - {description} (未安装)")
            missing_deps.append(module)
    
    if missing_deps:
        print(f"\n⚠️ 缺少依赖库:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print(f"\n安装命令:")
        if 'torch' in missing_deps:
            print("  pip install torch torchvision")
        if any(dep in missing_deps for dep in ['numpy', 'matplotlib', 'PIL']):
            remaining = [dep for dep in missing_deps if dep != 'torch']
            if remaining:
                print(f"  pip install {' '.join(remaining)}")
        return False
    
    return True

def main():
    """主函数"""
    print("🔍 Vision Transformer (ViT) 项目验证工具")
    print("这个脚本验证 ViT 项目结构的完整性")
    print()
    
    # 验证项目结构
    structure_ok = verify_vit_structure()
    
    # 检查依赖
    deps_ok = check_imports()
    
    if structure_ok and deps_ok:
        print("\n🎉 ViT 项目验证通过！")
        print("所有文件完整，依赖已安装，可以开始学习。")
        
        print("\n🏃‍♂️ 快速开始:")
        print("  python3 vit/quick_start.py        # 快速入门演示")
        print("  python3 vit/run_examples.py       # 交互式学习")
        
    elif structure_ok:
        print("\n⚠️ 项目结构完整，但缺少依赖库")
        print("请先安装必要的依赖库，然后再开始学习。")
        
    else:
        print("\n❌ 项目结构验证失败！")
        print("请检查缺失的文件。")
    
    # 显示项目总结
    show_vit_summary()
    
    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
