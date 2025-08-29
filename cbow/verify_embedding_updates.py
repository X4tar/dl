import torch
import torch.nn as nn

# 创建一个简单的embedding层
embedding = nn.Embedding(10, 5)

# 记录初始权重
initial_weight = embedding.weight.data.clone()
print("初始权重:")
print(initial_weight)

# 创建一个简单的损失和优化器
optimizer = torch.optim.SGD(embedding.parameters(), lr=0.1)
criterion = nn.MSELoss()

# 模拟一次训练步骤
input_ids = torch.tensor([1, 2, 3])
target = torch.randn(3, 5)  # 随机目标

# 前向传播
output = embedding(input_ids)
loss = criterion(output, target)

# 反向传播
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 检查权重是否改变
updated_weight = embedding.weight.data
print("\n更新后权重:")
print(updated_weight)

print("\n权重是否改变:")
print(torch.equal(initial_weight, updated_weight))  # 应该是False

print("\n权重变化量:")
print(torch.abs(updated_weight - initial_weight))
