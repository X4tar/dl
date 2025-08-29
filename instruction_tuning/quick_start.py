"""
🎯 指令微调 (Instruction Tuning) 快速入门
演示现代大语言模型的指令微调过程
"""

import torch
import sys
import os
from pathlib import Path

# 添加路径以导入其他模块
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from sft_trainer import SFTTrainer, InstructionDataset, SimpleTokenizer, create_sample_model

def print_banner():
    """打印欢迎横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🎯 指令微调快速入门                        ║
    ║               Instruction Tuning Quick Start                 ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  📝 SFT    🏆 奖励模型    🔄 RLHF    💬 对话优化           ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def explain_instruction_tuning():
    """解释指令微调的概念"""
    print("""
🎓 指令微调 (Instruction Tuning) 基础知识:

📚 核心概念:
   指令微调是让预训练的语言模型能够理解和遵循人类指令的关键技术。
   它将通用的语言模型转化为能够执行特定任务的助手。

🔄 技术流程:
   1. 📖 预训练模型 → 基础语言理解能力
   2. 📝 监督微调 (SFT) → 指令遵循能力  
   3. 🏆 奖励模型训练 → 质量评估能力
   4. 🔄 强化学习 (RLHF) → 人类偏好对齐

💡 关键技术:
   • SFT: 用指令-回答对训练模型
   • RM: 训练奖励模型评估回答质量
   • RLHF: 用人类反馈优化模型行为
   • DPO: 直接偏好优化，简化RLHF流程

🌟 应用价值:
   • 💬 对话系统 (ChatGPT, Claude)
   • 🤖 AI助手 (代码、写作、分析)
   • 📊 专业工具 (医疗、法律、教育)
   • 🎯 定制化应用 (企业专用助手)
    """)

def demonstrate_instruction_formats():
    """演示不同的指令格式"""
    print("\n🎯 指令格式演示:")
    print("=" * 60)
    
    formats = [
        {
            "name": "基础问答格式",
            "template": "Q: {instruction}\nA: {response}",
            "example": {
                "instruction": "什么是深度学习？",
                "response": "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的复杂模式。"
            }
        },
        {
            "name": "指令-回应格式",
            "template": "### Instruction:\n{instruction}\n\n### Response:\n{response}",
            "example": {
                "instruction": "写一个Python函数计算斐波那契数列",
                "response": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
            }
        },
        {
            "name": "对话格式",
            "template": "Human: {instruction}\n\nAssistant: {response}",
            "example": {
                "instruction": "你好，能帮我解释一下什么是Transformer吗？",
                "response": "你好！Transformer是一种基于注意力机制的神经网络架构，广泛应用于自然语言处理任务。"
            }
        }
    ]
    
    for i, fmt in enumerate(formats, 1):
        print(f"\n{i}. {fmt['name']}:")
        print(f"   模板: {fmt['template']}")
        print("   示例:")
        formatted = fmt['template'].format(**fmt['example'])
        for line in formatted.split('\n'):
            print(f"      {line}")

def demonstrate_sft_training():
    """演示监督微调训练过程"""
    print("\n🚀 监督微调 (SFT) 训练演示:")
    print("=" * 60)
    
    # 设备设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 使用设备: {device}")
    
    # 创建分词器
    print("\n📝 步骤 1: 准备分词器")
    tokenizer = SimpleTokenizer()
    print(f"   ✅ 创建字符级分词器，词汇表大小: {tokenizer.vocab_size}")
    
    # 创建模型
    print("\n🤖 步骤 2: 准备模型")
    model = create_sample_model(tokenizer.vocab_size)
    
    # 创建数据集
    print("\n📊 步骤 3: 准备数据集")
    dataset = InstructionDataset(
        data_path="dummy_path",
        tokenizer=tokenizer,
        max_length=128
    )
    print(f"   ✅ 数据集大小: {len(dataset)} 条指令-回答对")
    
    # 展示数据样例
    print(f"   📋 数据样例:")
    sample = dataset[0]
    print(f"      指令: {sample['instruction']}")
    print(f"      回答: {sample['response']}")
    
    # 创建训练器
    print("\n🎯 步骤 4: 创建训练器")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=1e-4
    )
    print(f"   ✅ 训练器配置完成")
    print(f"      学习率: 1e-4")
    print(f"      优化器: AdamW")
    print(f"      权重衰减: 0.01")
    
    # 快速训练演示
    print("\n⚡ 步骤 5: 快速训练演示 (3个epoch)")
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )
    
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"\n📚 Epoch {epoch + 1}/{num_epochs}")
        avg_loss = trainer.train_epoch(dataloader, epoch)
        
        # 生成示例
        print(f"\n🤖 训练后生成测试:")
        test_instruction = "请解释什么是人工智能"
        response = trainer.generate_response(test_instruction, max_length=30)
        print(f"   Q: {test_instruction}")
        print(f"   A: {response}")
    
    print(f"\n✅ 训练完成！最终平均损失: {avg_loss:.4f}")
    
    return trainer

def demonstrate_inference():
    """演示推理和生成"""
    print("\n💬 推理和生成演示:")
    print("=" * 60)
    
    # 创建训练好的模型 (这里使用简单演示)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = SimpleTokenizer()
    model = create_sample_model(tokenizer.vocab_size)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    # 测试不同类型的指令
    test_instructions = [
        "请解释什么是机器学习",
        "写一首关于秋天的诗",
        "如何学习编程？",
        "Python中如何定义函数？",
        "请总结深度学习的主要优点"
    ]
    
    print("🎯 多样化指令测试:")
    for i, instruction in enumerate(test_instructions, 1):
        print(f"\n{i}. 指令: {instruction}")
        response = trainer.generate_response(instruction, max_length=50)
        print(f"   回答: {response}")

def show_rlhf_concept():
    """展示RLHF概念"""
    print("\n🔄 人类反馈强化学习 (RLHF) 概念:")
    print("=" * 60)
    
    print("""
🎯 RLHF 核心思想:
   让AI模型学习人类的偏好，产生更符合人类期望的回答

📊 技术流程:
   1. 📝 SFT阶段: 基础指令遵循能力
      ├── 输入: 指令-回答对数据
      ├── 目标: 学会理解和回应指令
      └── 输出: 初步可用的指令模型

   2. 🏆 奖励模型(RM)训练:
      ├── 输入: 指令 + 多个候选回答 + 人类偏好排序
      ├── 目标: 学会评估回答质量
      └── 输出: 能打分的奖励模型

   3. 🔄 PPO强化学习:
      ├── 输入: 指令 + SFT模型 + 奖励模型
      ├── 目标: 最大化奖励分数
      └── 输出: 对齐人类偏好的最终模型

💡 关键技术点:
   • 🎯 偏好数据收集: 人类标注员对回答进行排序
   • 📊 奖励建模: 将人类偏好转化为可计算的分数
   • ⚖️ 平衡约束: 防止模型偏离原始分布太远
   • 🔄 迭代优化: 持续改进模型表现

🌟 应用效果:
   • ✅ 更有帮助的回答
   • ✅ 更诚实的表达
   • ✅ 更安全的行为
   • ✅ 更符合人类价值观
    """)

def show_modern_techniques():
    """展示现代指令微调技术"""
    print("\n🚀 现代指令微调技术:")
    print("=" * 60)
    
    techniques = [
        {
            "name": "DPO (Direct Preference Optimization)",
            "description": "直接偏好优化，无需训练奖励模型",
            "advantages": ["简化流程", "训练稳定", "效果显著"],
            "use_case": "Claude, Llama2-Chat 等模型"
        },
        {
            "name": "Constitutional AI",
            "description": "基于宪法原则的AI对齐方法",
            "advantages": ["价值观对齐", "自我批评", "行为规范"],
            "use_case": "Claude 系列模型的核心技术"
        },
        {
            "name": "Self-Instruct",
            "description": "模型自我生成指令数据进行训练",
            "advantages": ["数据扩充", "降低成本", "多样性提升"],
            "use_case": "Stanford Alpaca, Self-Instruct"
        },
        {
            "name": "LoRA Fine-tuning",
            "description": "低秩适应微调，高效参数更新",
            "advantages": ["参数效率", "训练快速", "资源节省"],
            "use_case": "个人/小团队微调大模型"
        }
    ]
    
    for i, tech in enumerate(techniques, 1):
        print(f"\n{i}. {tech['name']}:")
        print(f"   📝 描述: {tech['description']}")
        print(f"   ✨ 优势: {', '.join(tech['advantages'])}")
        print(f"   🎯 应用: {tech['use_case']}")

def interactive_demo():
    """交互式演示"""
    print("\n🎮 交互式指令微调演示:")
    print("=" * 60)
    
    print("""
💡 这里演示了指令微调的完整流程:

🔧 如果您想深入实践:
   1. 准备高质量的指令-回答数据
   2. 使用更大的预训练模型 (如 LLaMA, GPT)
   3. 配置更强的计算资源 (GPU集群)
   4. 实施完整的RLHF流程
   5. 进行充分的安全性测试

📚 推荐学习资源:
   • OpenAI InstructGPT 论文
   • Anthropic Constitutional AI 论文  
   • Stanford Alpaca 项目
   • HuggingFace RLHF 教程
   • DeepSpeed Chat 框架
    """)

def main():
    """主函数"""
    print_banner()
    
    explain_instruction_tuning()
    
    demonstrate_instruction_formats()
    
    print("\n" + "="*80)
    print("🎯 开始实际演示...")
    print("="*80)
    
    try:
        # 演示SFT训练
        trainer = demonstrate_sft_training()
        
        # 演示推理
        demonstrate_inference()
        
        # 展示RLHF概念
        show_rlhf_concept()
        
        # 展示现代技术
        show_modern_techniques()
        
        # 交互式演示
        interactive_demo()
        
        print(f"\n" + "="*80)
        print("🎉 指令微调快速入门完成！")
        print("="*80)
        
        print("""
🎓 您已经学习了:
   ✅ 指令微调的基本概念和重要性
   ✅ SFT (监督微调) 的完整训练流程
   ✅ 不同指令格式的使用方法
   ✅ RLHF 人类反馈强化学习原理
   ✅ 现代指令微调技术发展趋势

🚀 下一步建议:
   1. 深入学习 RLHF 相关论文
   2. 实践更大规模的模型微调
   3. 探索多模态指令微调技术
   4. 关注AI安全和对齐研究
        """)
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        print("💡 这通常是因为模型较简单或数据量较小导致的")
        print("   在实际应用中，请使用更大的模型和更多的数据")

if __name__ == "__main__":
    main()
