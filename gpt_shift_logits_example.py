"""
GPT模型中 shift_logits 操作的详细解释和示例
解析代码：shift_logits = logits[..., :-1, :].contiguous()
"""

import torch
import torch.nn.functional as F

def explain_shift_logits():
    """
    详细解释 shift_logits 操作的原理和作用
    """
    print("=" * 60)
    print("GPT模型中 shift_logits 操作详解")
    print("=" * 60)
    
    # 1. 模拟场景设置
    batch_size = 2
    seq_len = 5
    vocab_size = 10
    
    # 模拟输入序列（token ids）
    input_ids = torch.tensor([
        [1, 3, 7, 2, 9],  # 句子1: "我 喜欢 编程 。 <EOS>"
        [4, 8, 1, 6, 5]   # 句子2: "今天 天气 我 很好 ！"
    ])
    
    # 模拟模型输出的logits [batch_size, seq_len, vocab_size]
    # 每个位置输出对词汇表中所有词的概率分布
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    print(f"原始输入序列 input_ids:")
    print(f"形状: {input_ids.shape}")
    print(f"内容:\n{input_ids}")
    print()
    
    print(f"模型输出 logits:")
    print(f"形状: {logits.shape}")
    print(f"解释: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]")
    print()
    
    # 2. 核心操作：shift_logits
    print("=" * 40)
    print("核心操作分析")
    print("=" * 40)
    
    # 这就是我们要解释的关键代码
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    print(f"shift_logits = logits[..., :-1, :]")
    print(f"shift_logits 形状: {shift_logits.shape}")
    print(f"解释: 去掉最后一个时间步的logits")
    print()
    
    print(f"shift_labels = input_ids[..., 1:]")
    print(f"shift_labels 形状: {shift_labels.shape}")
    print(f"shift_labels 内容:\n{shift_labels}")
    print(f"解释: 去掉第一个token，作为预测目标")
    print()
    
    # 3. 详细解释为什么要这样做
    print("=" * 40)
    print("为什么需要shift操作？")
    print("=" * 40)
    
    print("语言模型的训练目标：给定前面的词，预测下一个词")
    print()
    
    print("原始序列分析:")
    for i in range(batch_size):
        print(f"句子 {i+1}: {input_ids[i].tolist()}")
        print("训练对应关系:")
        for j in range(seq_len-1):
            input_context = input_ids[i, :j+1].tolist()
            target_token = input_ids[i, j+1].item()
            print(f"  输入: {input_context} -> 预测目标: {target_token}")
        print()
    
    # 4. 对齐后的效果
    print("=" * 40)
    print("shift操作后的对齐效果")
    print("=" * 40)
    
    for i in range(batch_size):
        print(f"句子 {i+1} 的预测对齐:")
        for j in range(seq_len-1):
            # logits[i, j, :] 是在位置j时对下一个词的预测分布
            # shift_labels[i, j] 是实际的下一个词
            print(f"  位置{j}: logits[{i},{j},:] 预测 -> 目标: {shift_labels[i,j].item()}")
        print()

def demonstrate_loss_calculation():
    """
    演示完整的损失计算过程
    """
    print("=" * 60)
    print("完整的损失计算演示")
    print("=" * 60)
    
    # 模拟数据
    batch_size, seq_len, vocab_size = 2, 4, 6
    
    # 输入序列
    input_ids = torch.tensor([
        [1, 2, 3, 4],  # "我 爱 编程 。"
        [2, 1, 4, 5]   # "爱 我 。 ！"
    ])
    
    # 模拟logits (简化为较小的数值便于理解)
    logits = torch.tensor([
        [[0.1, 2.0, 0.3, 0.2, 0.1, 0.1],  # 位置0：预测下一个词
         [0.2, 0.1, 3.0, 0.4, 0.2, 0.1],  # 位置1：预测下一个词  
         [0.1, 0.2, 0.1, 4.0, 0.3, 0.2],  # 位置2：预测下一个词
         [0.3, 0.1, 0.2, 0.1, 5.0, 0.1]], # 位置3：预测下一个词
        [[0.2, 1.8, 0.4, 0.3, 0.2, 0.1],
         [2.1, 0.3, 0.2, 0.1, 0.2, 0.1],
         [0.1, 0.3, 0.2, 0.1, 4.2, 0.1],
         [0.2, 0.1, 0.3, 0.1, 0.2, 6.1]]
    ], dtype=torch.float32)
    
    print("输入序列:")
    print(input_ids)
    print()
    
    print("Logits形状:", logits.shape)
    print()
    
    # Shift操作
    shift_logits = logits[..., :-1, :].contiguous()  # [2, 3, 6]
    shift_labels = input_ids[..., 1:].contiguous()   # [2, 3]
    
    print("Shift后:")
    print(f"shift_logits形状: {shift_logits.shape}")
    print(f"shift_labels形状: {shift_labels.shape}")
    print(f"shift_labels内容:\n{shift_labels}")
    print()
    
    # 计算损失
    print("损失计算过程:")
    print("1. 将shift_logits reshape为 [batch_size * (seq_len-1), vocab_size]")
    print("2. 将shift_labels reshape为 [batch_size * (seq_len-1)]")
    print("3. 使用CrossEntropyLoss计算损失")
    print()
    
    # 重塑张量用于损失计算
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))  # [6, 6]
    flat_labels = shift_labels.view(-1)  # [6]
    
    print(f"展平后的logits形状: {flat_logits.shape}")
    print(f"展平后的labels形状: {flat_labels.shape}")
    print(f"展平后的labels: {flat_labels.tolist()}")
    print()
    
    # 计算交叉熵损失
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(flat_logits, flat_labels)
    
    print(f"最终损失: {loss.item():.4f}")
    
    # 详细显示每个预测
    print("\n详细预测分析:")
    probabilities = F.softmax(flat_logits, dim=-1)
    for i, (logit_row, prob_row, label) in enumerate(zip(flat_logits, probabilities, flat_labels)):
        predicted_token = torch.argmax(logit_row).item()
        confidence = prob_row[label].item()
        print(f"预测{i+1}: 预测词={predicted_token}, 实际词={label.item()}, 正确概率={confidence:.3f}")

def visualize_shift_concept():
    """
    可视化shift概念
    """
    print("=" * 60)
    print("Shift操作的可视化理解")
    print("=" * 60)
    
    # 示例句子："我 爱 编程"
    tokens = ["我", "爱", "编程", "<EOS>"]
    token_ids = [1, 2, 3, 4]
    
    print("原始序列:", " ".join(tokens))
    print("Token IDs:", token_ids)
    print()
    
    print("模型的预测任务:")
    print("┌" + "─" * 50 + "┐")
    print("│  输入上下文    │  预测目标  │  logits位置  │")
    print("├" + "─" * 50 + "┤")
    
    for i in range(len(tokens)-1):
        context = " ".join(tokens[:i+1])
        target = tokens[i+1]
        print(f"│  {context:<12} │  {target:<8} │  logits[{i}]   │")
    
    print("└" + "─" * 50 + "┘")
    print()
    
    print("Shift操作说明:")
    print("• shift_logits = logits[:-1, :] 取前3个位置的输出")
    print("• shift_labels = labels[1:]     取后3个token作为目标")
    print("• 这样就实现了 '输入→预测目标' 的完美对齐")

if __name__ == "__main__":
    # 运行所有示例
    explain_shift_logits()
    print("\n" + "="*60 + "\n")
    
    demonstrate_loss_calculation()
    print("\n" + "="*60 + "\n")
    
    visualize_shift_concept()
    
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    print("shift_logits = logits[..., :-1, :].contiguous() 的作用:")
    print("1. 去掉最后一个时间步的logits输出")
    print("2. 与去掉第一个token的labels对齐")  
    print("3. 实现'给定前文预测下一词'的训练目标")
    print("4. .contiguous()确保内存连续，提高计算效率")
    print("5. 这是所有自回归语言模型训练的标准操作")
