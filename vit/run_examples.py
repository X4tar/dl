"""
Vision Transformer (ViT) 交互式示例运行器
提供菜单式的学习体验
"""

import sys
import os

# 确保能导入本地模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def show_menu():
    """显示主菜单"""
    print("\n" + "=" * 60)
    print("🎯 Vision Transformer (ViT) 交互式学习")
    print("=" * 60)
    print("请选择要运行的示例:")
    print()
    print("基础组件:")
    print("  1. ViT 组件测试 (vit_components.py)")
    print("  2. 图像分块和嵌入 (patch_embedding.py)")
    print("  3. 完整 ViT 模型 (vit_model.py)")
    print()
    print("实践应用:")
    print("  4. 快速入门演示 (quick_start.py)")
    print("  5. 模型训练示例 (vit_trainer.py)")
    print("  6. 注意力可视化 (attention_visualization.py)")
    print()
    print("学习资料:")
    print("  7. 查看教程文档 (README)")
    print("  8. 模型架构对比")
    print("  9. ViT vs CNN 对比")
    print()
    print("  0. 退出")
    print("=" * 60)


def run_component_test():
    """运行组件测试"""
    print("\n🔧 运行 ViT 组件测试...")
    try:
        from vit_components import test_components, demonstrate_attention_mechanism
        
        print("正在测试基础组件...")
        test_components()
        
        print("\n正在演示注意力机制...")
        demonstrate_attention_mechanism()
        
        print("\n✅ 组件测试完成！")
        
    except Exception as e:
        print(f"❌ 组件测试失败: {e}")


def run_patch_embedding():
    """运行图像分块测试"""
    print("\n🖼️ 运行图像分块和嵌入演示...")
    try:
        from patch_embedding import demonstrate_embedding_process, analyze_patch_patterns
        
        print("正在演示嵌入过程...")
        demonstrate_embedding_process()
        
        print("\n正在分析不同 patch 大小...")
        analyze_patch_patterns()
        
        print("\n✅ 图像嵌入演示完成！")
        
    except Exception as e:
        print(f"❌ 图像嵌入演示失败: {e}")


def run_model_test():
    """运行模型测试"""
    print("\n🏗️ 运行完整 ViT 模型测试...")
    try:
        from vit_model import test_vit_models, demonstrate_model_components
        
        print("正在测试不同配置的模型...")
        test_vit_models()
        
        print("\n正在演示模型组件...")
        demonstrate_model_components()
        
        print("\n✅ 模型测试完成！")
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")


def run_quick_start():
    """运行快速入门"""
    print("\n🚀 运行快速入门演示...")
    try:
        from quick_start import main
        main()
        
    except Exception as e:
        print(f"❌ 快速入门失败: {e}")


def run_training():
    """运行训练示例"""
    print("\n🎯 运行训练示例...")
    print("注意: 这将创建一个小型数据集并训练模型")
    
    try:
        choice = input("确认开始训练? (y/N): ").lower()
        if choice != 'y':
            print("训练已取消")
            return
            
        from vit_trainer import demo_training
        demo_training()
        
        print("\n✅ 训练演示完成！")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")


def run_attention_visualization():
    """运行注意力可视化"""
    print("\n👁️ 运行注意力可视化...")
    try:
        import torch
        from vit_model import create_vit_model
        from quick_start import visualize_attention_pattern
        
        # 创建模型和测试数据
        model = create_vit_model('vit_tiny', num_classes=10)
        images = torch.randn(2, 3, 224, 224)
        
        # 获取注意力权重
        model.eval()
        with torch.no_grad():
            _, attention_weights = model(images, return_attention=True)
        
        # 可视化
        visualize_attention_pattern(model, images, attention_weights, save_plots=True)
        
        print("\n✅ 注意力可视化完成！")
        
    except Exception as e:
        print(f"❌ 注意力可视化失败: {e}")


def show_tutorial():
    """显示教程内容"""
    print("\n📚 ViT 教程概览")
    print("=" * 60)
    
    tutorial_content = """
🎯 Vision Transformer (ViT) 学习指南

📖 核心概念:
1. 图像分块 (Image Patching)
   - 将图像切分成固定大小的patches
   - 每个patch被视为一个token
   
2. 线性投影 (Linear Projection)
   - 将patch转换为嵌入向量
   - 使用卷积层实现高效投影
   
3. 位置编码 (Positional Encoding)
   - 为每个patch添加位置信息
   - 使用可学习的位置嵌入
   
4. CLS Token
   - 特殊的分类token
   - 用于聚合信息进行最终分类
   
5. Transformer 编码器
   - 多头自注意力机制
   - 前馈网络
   - 残差连接和层归一化

🚀 学习路径建议:
1. 从快速入门开始 (选项4)
2. 理解基础组件 (选项1)
3. 学习图像嵌入 (选项2)
4. 掌握完整模型 (选项3)
5. 实践训练过程 (选项5)
6. 分析注意力机制 (选项6)

💡 关键特点:
- 全局注意力: 每个patch都能关注到所有其他patches
- 可扩展性: 容易扩展到不同尺寸和任务
- 数据需求: 需要大量数据或预训练才能发挥最佳性能
- 归纳偏置: 相比CNN具有更少的归纳偏置
"""
    
    print(tutorial_content)
    
    print("\n📁 文件说明:")
    print("- README_vit_tutorial.md: 详细教程文档")
    print("- vit_components.py: 基础组件实现")
    print("- patch_embedding.py: 图像分块和嵌入")
    print("- vit_model.py: 完整模型实现")
    print("- vit_trainer.py: 训练和评估")
    print("- quick_start.py: 快速入门")
    print("- run_examples.py: 本交互式运行器")


def compare_architectures():
    """比较不同架构"""
    print("\n🔍 ViT 模型架构对比")
    print("=" * 60)
    
    try:
        from vit_model import create_vit_model, count_parameters
        
        architectures = {
            'ViT-Tiny': {'model': 'vit_tiny', 'embed_dim': 192, 'layers': 12, 'heads': 3},
            'ViT-Small': {'model': 'vit_small', 'embed_dim': 384, 'layers': 12, 'heads': 6},
            'ViT-Base': {'model': 'vit_base', 'embed_dim': 768, 'layers': 12, 'heads': 12},
            'ViT-Large': {'model': 'vit_large', 'embed_dim': 1024, 'layers': 24, 'heads': 16},
            'ViT-Huge': {'model': 'vit_huge', 'embed_dim': 1280, 'layers': 32, 'heads': 16},
        }
        
        print(f"{'架构':<12} {'嵌入维度':<8} {'层数':<6} {'头数':<6} {'参数数量':<12} {'相对大小':<8}")
        print("-" * 70)
        
        tiny_params = None
        
        for arch_name, config in architectures.items():
            try:
                model = create_vit_model(model_name=config['model'], num_classes=1000)
                params, _ = count_parameters(model)
                
                if tiny_params is None:
                    tiny_params = params
                
                relative_size = params / tiny_params
                
                print(f"{arch_name:<12} {config['embed_dim']:<8} {config['layers']:<6} "
                      f"{config['heads']:<6} {params:<12,} {relative_size:<8.1f}x")
                
            except Exception as e:
                print(f"{arch_name:<12} 无法创建: {e}")
        
        print("\n💡 选择建议:")
        print("- ViT-Tiny: 学习和实验，资源有限的环境")
        print("- ViT-Small: 中等规模任务，平衡性能和效率")
        print("- ViT-Base: 标准选择，大多数任务的起点")
        print("- ViT-Large: 大规模数据集，追求更高性能")
        print("- ViT-Huge: 最大规模任务，顶级性能需求")
        
    except Exception as e:
        print(f"架构比较失败: {e}")


def compare_vit_cnn():
    """比较ViT和CNN"""
    print("\n⚖️ ViT vs CNN 对比")
    print("=" * 60)
    
    comparison = """
📊 特性对比:

┌─────────────────┬─────────────────┬─────────────────┐
│      特性       │       ViT       │       CNN       │
├─────────────────┼─────────────────┼─────────────────┤
│   归纳偏置      │      较少       │      较强       │
│   局部性偏置    │      无         │      强         │
│   平移不变性    │    学习获得     │      内建       │
│   全局感受野    │    第一层起     │    逐层增长     │
│   数据需求      │      大量       │      中等       │
│   计算复杂度    │   O(n²) 注意力  │  O(n) 卷积      │
│   可解释性      │   注意力图      │    特征图       │
│   预训练效果    │      显著       │      中等       │
└─────────────────┴─────────────────┴─────────────────┘

🎯 适用场景:

ViT 更适合:
✅ 大规模数据集 (ImageNet-21K等)
✅ 需要全局建模的任务
✅ 有充足预训练资源
✅ 追求最新架构的研究
✅ 多模态任务 (视觉+语言)

CNN 更适合:
✅ 中小规模数据集
✅ 资源受限的环境
✅ 需要强局部性偏置的任务
✅ 实时推理要求
✅ 从头训练的场景

🔄 混合方法:
- ConvNext: CNN架构 + Transformer设计原则
- Swin Transformer: 层次化ViT + 局部窗口注意力
- PVT: 金字塔式ViT + 多尺度特征
- CvT: 卷积 + Transformer 的混合架构

📈 性能趋势:
1. 小数据集: CNN > ViT
2. 大数据集: ViT ≥ CNN  
3. 预训练场景: ViT > CNN
4. 效率要求: CNN > ViT
"""
    
    print(comparison)


def main():
    """主函数"""
    while True:
        show_menu()
        
        try:
            choice = input("\n请输入选项 (0-9): ").strip()
            
            if choice == '0':
                print("\n👋 感谢使用 ViT 学习系统！")
                break
            elif choice == '1':
                run_component_test()
            elif choice == '2':
                run_patch_embedding()
            elif choice == '3':
                run_model_test()
            elif choice == '4':
                run_quick_start()
            elif choice == '5':
                run_training()
            elif choice == '6':
                run_attention_visualization()
            elif choice == '7':
                show_tutorial()
            elif choice == '8':
                compare_architectures()
            elif choice == '9':
                compare_vit_cnn()
            else:
                print("\n❌ 无效选项，请重新选择")
            
            if choice != '0':
                input("\n按回车键继续...")
                
        except KeyboardInterrupt:
            print("\n\n👋 程序已中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            input("按回车键继续...")


if __name__ == "__main__":
    main()
