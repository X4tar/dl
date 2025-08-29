"""
BERT 运行示例集合
包含所有 BERT 相关的演示和测试
"""

import sys
import traceback

def run_bert_components():
    """运行 BERT 组件测试"""
    print("🧩 运行 BERT 组件测试")
    print("=" * 50)
    
    try:
        from bert_components import test_bert_components, demonstrate_bert_attention
        
        # 测试基础组件
        print("1. 测试 BERT 基础组件...")
        test_bert_components()
        
        # 演示注意力机制
        print("\n2. 演示 BERT 注意力机制...")
        demonstrate_bert_attention()
        
        print("✅ BERT 组件测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ BERT 组件测试失败: {str(e)}")
        print("详细错误信息:")
        traceback.print_exc()
        return False


def run_bert_models():
    """运行 BERT 模型测试"""
    print("\n🤖 运行 BERT 模型测试")
    print("=" * 50)
    
    try:
        from bert_model import test_bert_models
        
        print("测试各种 BERT 模型变体...")
        test_bert_models()
        
        print("✅ BERT 模型测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ BERT 模型测试失败: {str(e)}")
        print("详细错误信息:")
        traceback.print_exc()
        return False


def run_bert_pretraining():
    """运行 BERT 预训练演示"""
    print("\n🎓 运行 BERT 预训练演示")
    print("=" * 50)
    
    try:
        from bert_pretraining import (
            demonstrate_bert_pretraining, analyze_bert_attention,
            demonstrate_mlm_task, demonstrate_nsp_task
        )
        
        print("1. 演示 BERT 预训练过程...")
        demonstrate_bert_pretraining()
        
        print("\n2. 分析 BERT 注意力模式...")
        analyze_bert_attention()
        
        print("\n3. 演示 MLM 任务...")
        demonstrate_mlm_task()
        
        print("\n4. 演示 NSP 任务...")
        demonstrate_nsp_task()
        
        print("✅ BERT 预训练演示完成!")
        return True
        
    except Exception as e:
        print(f"❌ BERT 预训练演示失败: {str(e)}")
        print("详细错误信息:")
        traceback.print_exc()
        return False


def run_bert_quick_start():
    """运行 BERT 快速入门"""
    print("\n🚀 运行 BERT 快速入门")
    print("=" * 50)
    
    try:
        from bert.quick_start import (
            quick_demo, compare_bert_variants, demonstrate_key_differences
        )
        
        print("1. 快速演示 BERT 功能...")
        quick_demo()
        
        print("\n2. 比较 BERT 模型变体...")
        compare_bert_variants()
        
        print("\n3. 演示 BERT 关键特性...")
        demonstrate_key_differences()
        
        print("✅ BERT 快速入门完成!")
        return True
        
    except Exception as e:
        print(f"❌ BERT 快速入门失败: {str(e)}")
        print("详细错误信息:")
        traceback.print_exc()
        return False


def run_comprehensive_bert_tutorial():
    """运行完整的 BERT 教程"""
    print("🎯 BERT 完整教程演示")
    print("=" * 60)
    
    print("📚 本教程将带您完整了解 BERT:")
    print("1. 🧩 核心组件实现")
    print("2. 🤖 模型架构详解")  
    print("3. 🎓 预训练过程演示")
    print("4. 🚀 快速使用指南")
    print("5. 📊 性能分析对比")
    
    results = []
    
    # 1. 组件测试
    print("\n" + "="*60)
    print("第1部分: BERT 核心组件")
    print("="*60)
    results.append(run_bert_components())
    
    # 2. 模型测试
    print("\n" + "="*60)
    print("第2部分: BERT 模型架构")
    print("="*60)
    results.append(run_bert_models())
    
    # 3. 预训练演示
    print("\n" + "="*60)
    print("第3部分: BERT 预训练过程")
    print("="*60)
    results.append(run_bert_pretraining())
    
    # 4. 快速入门
    print("\n" + "="*60)
    print("第4部分: BERT 快速使用")
    print("="*60)
    results.append(run_bert_quick_start())
    
    # 总结结果
    print("\n" + "="*60)
    print("🎉 BERT 完整教程总结")
    print("="*60)
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"✅ 成功完成: {success_count}/{total_count} 个部分")
    
    if success_count == total_count:
        print("🌟 恭喜！您已经完成了完整的 BERT 学习教程!")
        print("\n📖 学习成果:")
        print("- ✅ 理解了 BERT 的核心组件实现")
        print("- ✅ 掌握了 BERT 的模型架构")
        print("- ✅ 了解了 MLM 和 NSP 预训练任务")
        print("- ✅ 学会了 BERT 在不同任务中的应用")
        print("- ✅ 分析了 BERT 的注意力机制")
        
        print("\n🎓 下一步建议:")
        print("1. 尝试在真实数据集上微调 BERT")
        print("2. 研究 BERT 的改进版本 (RoBERTa, ALBERT, DeBERTa)")
        print("3. 探索 BERT 在多语言任务中的应用")
        print("4. 学习 GPT 等生成式预训练模型")
        
    else:
        failed_parts = []
        part_names = ["核心组件", "模型架构", "预训练过程", "快速使用"]
        for i, success in enumerate(results):
            if not success:
                failed_parts.append(part_names[i])
        
        print(f"❌ 以下部分执行失败: {', '.join(failed_parts)}")
        print("建议检查代码实现和依赖关系")
    
    return success_count == total_count


def show_bert_architecture():
    """展示 BERT 架构图（文本版）"""
    print("\n🏗️ BERT 架构图 (文本版)")
    print("=" * 60)
    
    architecture = """
    输入层
    ┌─────────────────────────────────────────┐
    │ Token Embeddings + Segment Embeddings   │
    │        + Position Embeddings            │
    └─────────────────────────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │           Transformer Layer 1           │
    │  ┌───────────────┐ ┌─────────────────┐  │
    │  │ Multi-Head    │ │   Feed Forward  │  │
    │  │  Attention    │ │    Network      │  │
    │  └───────────────┘ └─────────────────┘  │
    └─────────────────────────────────────────┘
                        │
                        ▼
                      ...
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │           Transformer Layer N           │
    │  ┌───────────────┐ ┌─────────────────┐  │
    │  │ Multi-Head    │ │   Feed Forward  │  │
    │  │  Attention    │ │    Network      │  │
    │  └───────────────┘ └─────────────────┘  │
    └─────────────────────────────────────────┘
                        │
                        ▼
    输出层
    ┌─────────────────────────────────────────┐
    │  Sequence Output    │   Pooled Output   │
    │ [batch, seq, dim]   │  [batch, dim]     │
    └─────────────────────────────────────────┘
    """
    
    print(architecture)
    
    print("📋 BERT 关键特点:")
    print("1. 双向编码: 同时考虑左右上下文")
    print("2. 预训练任务: MLM + NSP")
    print("3. 迁移学习: 预训练 + 微调")
    print("4. 多任务适配: 分类、标注、问答等")


def interactive_bert_demo():
    """交互式 BERT 演示"""
    print("\n🎮 交互式 BERT 演示")
    print("=" * 50)
    
    while True:
        print("\n请选择要运行的演示:")
        print("1. BERT 组件测试")
        print("2. BERT 模型测试")
        print("3. BERT 预训练演示")
        print("4. BERT 快速入门")
        print("5. 完整教程")
        print("6. 架构图展示")
        print("0. 退出")
        
        try:
            choice = input("\n请输入选项 (0-6): ").strip()
            
            if choice == '0':
                print("👋 再见！感谢使用 BERT 演示系统!")
                break
            elif choice == '1':
                run_bert_components()
            elif choice == '2':
                run_bert_models()
            elif choice == '3':
                run_bert_pretraining()
            elif choice == '4':
                run_bert_quick_start()
            elif choice == '5':
                run_comprehensive_bert_tutorial()
            elif choice == '6':
                show_bert_architecture()
            else:
                print("❌ 无效选项，请重新输入!")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出演示系统!")
            break
        except Exception as e:
            print(f"❌ 执行过程中发生错误: {str(e)}")


if __name__ == "__main__":
    print("🤖 BERT 演示系统")
    print("=" * 60)
    print("欢迎使用 BERT (Bidirectional Encoder Representations from Transformers) 演示系统!")
    print("这是一个完整的 BERT 学习和实践平台。")
    
    if len(sys.argv) > 1:
        # 命令行参数模式
        mode = sys.argv[1].lower()
        
        if mode == 'components':
            run_bert_components()
        elif mode == 'models':
            run_bert_models()
        elif mode == 'pretraining':
            run_bert_pretraining()
        elif mode == 'quickstart':
            run_bert_quick_start()
        elif mode == 'tutorial':
            run_comprehensive_bert_tutorial()
        elif mode == 'architecture':
            show_bert_architecture()
        else:
            print(f"❌ 未知模式: {mode}")
            print("可用模式: components, models, pretraining, quickstart, tutorial, architecture")
    else:
        # 交互模式
        interactive_bert_demo()
