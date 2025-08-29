#!/usr/bin/env python3
"""
🔍 项目完整性验证脚本
验证所有模块是否正确实现并可以正常运行
"""

import os
import sys
import importlib.util
from pathlib import Path

def check_file_exists(filepath):
    """检查文件是否存在"""
    return os.path.exists(filepath)

def check_module_import(module_path, module_name):
    """检查模块是否可以正常导入"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, "✅"
    except Exception as e:
        return False, f"❌ {str(e)[:50]}..."

def validate_transformer_module():
    """验证 Transformer 模块"""
    print("🔄 验证 Transformer 模块")
    print("-" * 50)
    
    files_to_check = [
        "transfomer/README_transformer_tutorial.md",
        "transfomer/transformer_components.py",
        "transfomer/positional_encoding.py", 
        "transfomer/transformer_model.py",
        "transfomer/train_transformer.py",
        "transfomer/attention_visualization.py",
        "transfomer/text_generation_example.py",
        "transfomer/quick_start.py",
        "transfomer/run_examples.py",
        "transfomer/verify_structure.py"
    ]
    
    results = []
    for file_path in files_to_check:
        exists = check_file_exists(file_path)
        status = "✅" if exists else "❌"
        results.append((file_path, exists))
        print(f"{status} {file_path}")
    
    # 测试关键模块导入
    print("\n📦 测试模块导入:")
    key_modules = [
        ("transfomer/transformer_components.py", "transformer_components"),
        ("transfomer/positional_encoding.py", "positional_encoding"),
        ("transfomer/transformer_model.py", "transformer_model")
    ]
    
    for module_path, module_name in key_modules:
        if check_file_exists(module_path):
            can_import, status = check_module_import(module_path, module_name)
            print(f"{status} {module_name}")
            results.append((module_name, can_import))
    
    return results

def validate_vit_module():
    """验证 ViT 模块"""
    print("\n🖼️ 验证 Vision Transformer 模块")
    print("-" * 50)
    
    files_to_check = [
        "vit/README_vit_tutorial.md",
        "vit/vit_components.py",
        "vit/patch_embedding.py",
        "vit/vit_model.py",
        "vit/vit_trainer.py",
        "vit/quick_start.py",
        "vit/run_examples.py",
        "vit/verify_vit_structure.py"
    ]
    
    results = []
    for file_path in files_to_check:
        exists = check_file_exists(file_path)
        status = "✅" if exists else "❌"
        results.append((file_path, exists))
        print(f"{status} {file_path}")
    
    # 测试关键模块导入
    print("\n📦 测试模块导入:")
    key_modules = [
        ("vit/vit_components.py", "vit_components"),
        ("vit/patch_embedding.py", "patch_embedding"),
        ("vit/vit_model.py", "vit_model")
    ]
    
    for module_path, module_name in key_modules:
        if check_file_exists(module_path):
            can_import, status = check_module_import(module_path, module_name)
            print(f"{status} {module_name}")
            results.append((module_name, can_import))
    
    return results

def validate_bert_module():
    """验证 BERT 模块"""
    print("\n🤖 验证 BERT 模块")
    print("-" * 50)
    
    files_to_check = [
        "bert/README_bert_tutorial.md",
        "bert/bert_components.py",
        "bert/bert_model.py",
        "bert/bert_pretraining.py",
        "bert/quick_start.py",
        "bert/run_examples.py"
    ]
    
    results = []
    for file_path in files_to_check:
        exists = check_file_exists(file_path)
        status = "✅" if exists else "❌"
        results.append((file_path, exists))
        print(f"{status} {file_path}")
    
    # 测试关键模块导入
    print("\n📦 测试模块导入:")
    key_modules = [
        ("bert/bert_components.py", "bert_components"),
        ("bert/bert_model.py", "bert_model"),
        ("bert/bert_pretraining.py", "bert_pretraining")
    ]
    
    for module_path, module_name in key_modules:
        if check_file_exists(module_path):
            can_import, status = check_module_import(module_path, module_name)
            print(f"{status} {module_name}")
            results.append((module_name, can_import))
    
    return results

def validate_cbow_module():
    """验证 CBOW 对比模块"""
    print("\n📝 验证 CBOW 对比模块")
    print("-" * 50)
    
    files_to_check = [
        "cbow/cbow_example.py",
        "cbow/cbow_model.py",
        "cbow/test_cbow.py",
        "cbow/verify_embedding_updates.py"
    ]
    
    results = []
    for file_path in files_to_check:
        exists = check_file_exists(file_path)
        status = "✅" if exists else "❌"
        results.append((file_path, exists))
        print(f"{status} {file_path}")
    
    return results

def validate_project_structure():
    """验证项目整体结构"""
    print("\n📁 验证项目整体结构")
    print("-" * 50)
    
    core_files = [
        "README.md",
        "demo_complete_project.py",
        "总结_完整Transformer教学项目.md",
        "pyproject.toml"
    ]
    
    results = []
    for file_path in core_files:
        exists = check_file_exists(file_path)
        status = "✅" if exists else "❌"
        results.append((file_path, exists))
        print(f"{status} {file_path}")
    
    # 检查目录结构
    directories = ["transfomer", "vit", "bert", "cbow"]
    print(f"\n📂 检查目录结构:")
    for directory in directories:
        exists = os.path.isdir(directory)
        status = "✅" if exists else "❌"
        results.append((directory, exists))
        print(f"{status} {directory}/")
    
    return results

def check_dependencies():
    """检查依赖包"""
    print("\n📦 检查关键依赖")
    print("-" * 50)
    
    dependencies = [
        "torch",
        "numpy", 
        "matplotlib",
        "seaborn"
    ]
    
    results = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}")
            results.append((dep, True))
        except ImportError:
            print(f"❌ {dep} (未安装)")
            results.append((dep, False))
    
    return results

def generate_summary(all_results):
    """生成验证总结"""
    print("\n" + "="*60)
    print("📊 验证结果总结")
    print("="*60)
    
    total_items = 0
    passed_items = 0
    
    for module_name, results in all_results.items():
        module_total = len(results)
        module_passed = sum(1 for _, success in results if success)
        
        total_items += module_total
        passed_items += module_passed
        
        success_rate = (module_passed / module_total * 100) if module_total > 0 else 0
        print(f"{module_name}: {module_passed}/{module_total} ({success_rate:.1f}%)")
    
    overall_rate = (passed_items / total_items * 100) if total_items > 0 else 0
    
    print(f"\n🎯 总体通过率: {passed_items}/{total_items} ({overall_rate:.1f}%)")
    
    if overall_rate >= 90:
        print("🎉 恭喜！项目验证基本通过！")
        print("✨ 所有核心模块都已正确实现")
    elif overall_rate >= 70:
        print("⚠️ 项目大部分功能正常，但有些问题需要检查")
    else:
        print("❌ 项目存在较多问题，需要仔细检查实现")
    
    return overall_rate

def main():
    """主验证函数"""
    print("🔍 " + "="*50)
    print("   Transformer 教学项目完整性验证")
    print("="*55)
    
    print("\n🎯 开始验证项目各个模块...")
    
    # 验证各个模块
    all_results = {}
    
    # 验证项目结构
    all_results["项目结构"] = validate_project_structure()
    
    # 验证依赖
    all_results["依赖包"] = check_dependencies()
    
    # 验证各个功能模块
    all_results["Transformer"] = validate_transformer_module()
    all_results["ViT"] = validate_vit_module() 
    all_results["BERT"] = validate_bert_module()
    all_results["CBOW"] = validate_cbow_module()
    
    # 生成总结
    overall_rate = generate_summary(all_results)
    
    # 给出建议
    print("\n💡 建议下一步:")
    if overall_rate >= 90:
        print("1. 🚀 运行 demo_complete_project.py 体验完整功能")
        print("2. 📚 阅读各模块的 README 深入学习")
        print("3. 🎯 尝试训练自己的模型")
        print("4. 💡 基于此项目实现新的模型变体")
    else:
        print("1. 🔧 检查缺失的文件和模块")
        print("2. 📦 安装缺少的依赖包：pip install torch numpy matplotlib seaborn")
        print("3. 🐛 查看具体的错误信息并修复")
        print("4. 🆘 如需帮助，请查看项目文档或提交 Issue")
    
    print(f"\n✨ 验证完成！项目完整度: {overall_rate:.1f}%")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 验证被用户中断")
    except Exception as e:
        print(f"\n❌ 验证过程中出错: {e}")
        print("💡 请检查项目完整性")
