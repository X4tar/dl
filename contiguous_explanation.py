"""
深入解析 .contiguous() 的使用场景和必要性
什么时候需要手动调用 .contiguous()？
"""

import torch
import time

def demonstrate_tensor_memory_layout():
    """
    演示张量的内存布局和连续性
    """
    print("=" * 60)
    print("张量内存布局和连续性详解")
    print("=" * 60)
    
    # 创建一个连续的张量
    x = torch.randn(2, 3, 4)
    print(f"原始张量 x:")
    print(f"形状: {x.shape}")
    print(f"是否连续: {x.is_contiguous()}")
    print(f"内存步长: {x.stride()}")
    print()
    
    # 转置操作会改变内存布局
    y = x.transpose(1, 2)  # 交换维度1和2
    print(f"转置后的张量 y = x.transpose(1, 2):")
    print(f"形状: {y.shape}")
    print(f"是否连续: {y.is_contiguous()}")  # False!
    print(f"内存步长: {y.stride()}")
    print()
    
    # 使用 .contiguous() 重新整理内存
    z = y.contiguous()
    print(f"调用 .contiguous() 后的张量 z:")
    print(f"形状: {z.shape}")
    print(f"是否连续: {z.is_contiguous()}")  # True!
    print(f"内存步长: {z.stride()}")
    print()

def demonstrate_view_vs_contiguous():
    """
    演示为什么某些操作需要连续内存
    """
    print("=" * 60)
    print("为什么需要连续内存？.view() 操作的要求")
    print("=" * 60)
    
    # 创建张量
    x = torch.randn(2, 3, 4)
    print(f"原始张量 x: {x.shape}")
    
    # 直接 view 是可以的（连续内存）
    try:
        reshaped = x.view(-1)  # 展平
        print(f"x.view(-1) 成功: {reshaped.shape}")
    except Exception as e:
        print(f"x.view(-1) 失败: {e}")
    print()
    
    # 转置后再 view
    y = x.transpose(1, 2)
    print(f"转置后 y: {y.shape}, 连续性: {y.is_contiguous()}")
    
    try:
        reshaped = y.view(-1)  # 这会失败！
        print(f"y.view(-1) 成功: {reshaped.shape}")
    except Exception as e:
        print(f"y.view(-1) 失败: {e}")
    print()
    
    # 先调用 .contiguous() 再 view
    try:
        reshaped = y.contiguous().view(-1)
        print(f"y.contiguous().view(-1) 成功: {reshaped.shape}")
    except Exception as e:
        print(f"y.contiguous().view(-1) 失败: {e}")
    print()

def demonstrate_performance_impact():
    """
    演示连续性对性能的影响
    """
    print("=" * 60)
    print("连续性对性能的影响")
    print("=" * 60)
    
    # 创建大张量
    size = 1000
    x = torch.randn(size, size)
    
    # 连续张量的操作时间
    start_time = time.time()
    for _ in range(100):
        result = torch.sum(x)
    continuous_time = time.time() - start_time
    
    # 非连续张量的操作时间
    y = x.transpose(0, 1)  # 非连续
    start_time = time.time()
    for _ in range(100):
        result = torch.sum(y)
    non_continuous_time = time.time() - start_time
    
    # 手动调用 contiguous 后的时间
    z = y.contiguous()
    start_time = time.time()
    for _ in range(100):
        result = torch.sum(z)
    manual_continuous_time = time.time() - start_time
    
    print(f"连续张量操作时间: {continuous_time:.4f}s")
    print(f"非连续张量操作时间: {non_continuous_time:.4f}s") 
    print(f"手动连续化后操作时间: {manual_continuous_time:.4f}s")
    print(f"性能差异: {non_continuous_time/continuous_time:.2f}x")
    print()

def common_scenarios_requiring_contiguous():
    """
    需要调用 .contiguous() 的常见场景
    """
    print("=" * 60)
    print("常见的需要 .contiguous() 的场景")
    print("=" * 60)
    
    print("1. 转置操作后使用 .view()")
    x = torch.randn(2, 3, 4)
    y = x.transpose(1, 2)
    print(f"   x.transpose(1, 2).is_contiguous(): {y.is_contiguous()}")
    print("   需要: y.contiguous().view(-1)")
    print()
    
    print("2. 切片操作产生不连续张量")
    x = torch.randn(10, 20, 30)
    y = x[:, ::2, :]  # 隔一个取一个
    print(f"   x[:, ::2, :].is_contiguous(): {y.is_contiguous()}")
    print("   某些操作可能需要: y.contiguous()")
    print()
    
    print("3. permute 操作后")
    x = torch.randn(2, 3, 4, 5)
    y = x.permute(0, 3, 1, 2)  # 重排维度
    print(f"   x.permute(0, 3, 1, 2).is_contiguous(): {y.is_contiguous()}")
    print("   可能需要: y.contiguous()")
    print()
    
    print("4. 某些神经网络操作")
    x = torch.randn(2, 3, 4, 5)
    y = x.transpose(-2, -1)
    print(f"   注意力机制中常见: x.transpose(-2, -1).is_contiguous(): {y.is_contiguous()}")
    print("   在reshape前需要: y.contiguous()")
    print()

def gpt_specific_example():
    """
    GPT模型中的具体例子
    """
    print("=" * 60)
    print("GPT模型中的 .contiguous() 使用")
    print("=" * 60)
    
    # 模拟 GPT 中的场景
    batch_size, seq_len, vocab_size = 2, 5, 1000
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    print("原始场景:")
    print(f"logits 形状: {logits.shape}")
    print(f"logits 连续性: {logits.is_contiguous()}")
    print()
    
    # 执行切片操作（这是 GPT 中的实际操作）
    shift_logits = logits[..., :-1, :]
    print("切片后:")
    print(f"shift_logits 形状: {shift_logits.shape}")
    print(f"shift_logits 连续性: {shift_logits.is_contiguous()}")
    print()
    
    print("为什么需要 .contiguous()？")
    
    # 尝试 view 操作（用于损失计算）
    try:
        flattened = shift_logits.view(-1, vocab_size)
        print(f"✓ shift_logits.view(-1, vocab_size) 成功: {flattened.shape}")
    except Exception as e:
        print(f"✗ shift_logits.view(-1, vocab_size) 失败: {e}")
        print("  这时就需要先调用 .contiguous()")
        flattened = shift_logits.contiguous().view(-1, vocab_size)
        print(f"✓ shift_logits.contiguous().view(-1, vocab_size) 成功: {flattened.shape}")
    print()
    
    print("实际情况分析:")
    print("- 在大多数情况下，简单的切片操作 [..., :-1, :] 通常保持连续性")
    print("- 但在某些复杂的张量操作序列后，可能会失去连续性")
    print("- 添加 .contiguous() 是一种防御性编程，确保后续操作不会失败")
    print("- 如果张量已经连续，.contiguous() 不会复制数据，开销很小")

def when_to_use_contiguous():
    """
    什么时候需要使用 .contiguous()
    """
    print("=" * 60)
    print("何时需要手动调用 .contiguous()？")
    print("=" * 60)
    
    scenarios = [
        {
            "situation": "使用 .view() 重塑张量时",
            "reason": ".view() 要求张量在内存中连续",
            "example": "x.transpose(0, 1).contiguous().view(-1)"
        },
        {
            "situation": "传递给C++扩展或CUDA核函数时", 
            "reason": "底层实现通常假设连续内存布局",
            "example": "custom_cuda_kernel(tensor.contiguous())"
        },
        {
            "situation": "某些PyTorch函数要求连续张量时",
            "reason": "函数内部使用了需要连续内存的操作",
            "example": "F.cross_entropy(logits.contiguous(), labels)"
        },
        {
            "situation": "性能优化需要时",
            "reason": "连续内存访问更高效",
            "example": "large_tensor.contiguous() # 在大量计算前"
        },
        {
            "situation": "避免潜在错误时",
            "reason": "防御性编程，确保操作成功",
            "example": "shift_logits.contiguous().view(-1, vocab_size)"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['situation']}")
        print(f"   原因: {scenario['reason']}")
        print(f"   示例: {scenario['example']}")
        print()
    
    print("经验法则:")
    print("✓ 如果不确定，添加 .contiguous() 通常是安全的")
    print("✓ 在使用 .view() 前，特别是在转置/切片后")
    print("✓ 在性能关键路径上，确保内存连续性")
    print("✓ 当遇到 'tensor is not contiguous' 错误时")
    print("✗ 不要在每个操作后都调用（不必要的开销）")

def demonstrate_zero_cost_contiguous():
    """
    演示 .contiguous() 的零成本情况
    """
    print("=" * 60)
    print(".contiguous() 的开销分析")
    print("=" * 60)
    
    # 已经连续的张量
    x = torch.randn(1000, 1000)
    print(f"张量是否连续: {x.is_contiguous()}")
    
    # 测试调用 .contiguous() 的开销
    start_time = time.time()
    for _ in range(1000):
        y = x.contiguous()  # 对已连续张量调用
    time_on_continuous = time.time() - start_time
    
    # 测试非连续张量的 .contiguous() 开销
    x_non_cont = x.transpose(0, 1)
    print(f"转置后是否连续: {x_non_cont.is_contiguous()}")
    
    start_time = time.time() 
    for _ in range(1000):
        y = x_non_cont.contiguous()  # 需要重新排列内存
    time_on_non_continuous = time.time() - start_time
    
    print(f"连续张量调用 .contiguous() 时间: {time_on_continuous:.4f}s")
    print(f"非连续张量调用 .contiguous() 时间: {time_on_non_continuous:.4f}s")
    print(f"开销比例: {time_on_non_continuous/time_on_continuous:.1f}x")
    print()
    print("结论:")
    print("- 对已连续张量调用 .contiguous() 几乎无开销")
    print("- 只有在需要重新排列内存时才有明显开销")
    print("- 这就是为什么可以放心地添加 .contiguous() 作为防御性编程")

if __name__ == "__main__":
    demonstrate_tensor_memory_layout()
    print("\n")
    
    demonstrate_view_vs_contiguous()
    print("\n")
    
    demonstrate_performance_impact()
    print("\n")
    
    common_scenarios_requiring_contiguous()
    print("\n")
    
    gpt_specific_example()
    print("\n")
    
    when_to_use_contiguous()
    print("\n")
    
    demonstrate_zero_cost_contiguous()
    
    print("\n" + "="*60)
    print("总结：什么时候需要手动调用 .contiguous()")
    print("="*60)
    print("1. 在使用 .view() 重塑张量前（特别是转置/切片后）")
    print("2. 传递给需要连续内存的底层函数时")
    print("3. 在性能关键代码中确保最佳内存访问模式")
    print("4. 作为防御性编程避免潜在的连续性错误")
    print("5. 当遇到 'tensor is not contiguous' 错误时")
    print("\n好消息：对已连续张量调用 .contiguous() 几乎无开销！")
