# ⚡ 模型优化与部署教程

> **从训练到生产：大模型优化和部署的完整指南**

## 🎯 概述

大模型的优化和部署是从研究到生产的关键环节，涉及模型压缩、推理加速、内存优化等多个方面。

## 📚 核心优化技术

### 🗜️ 模型压缩技术

#### 1. 量化 (Quantization)
```python
# INT8量化示例
def quantize_model(model):
    # FP32 → INT8，减少4倍内存
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    return quantized_model
```

**优势**:
- 📉 内存占用减少 50-75%
- ⚡ 推理速度提升 2-4x
- 💰 部署成本大幅降低

**类型对比**:
| 量化类型 | 精度损失 | 压缩比 | 适用场景 |
|----------|----------|--------|----------|
| INT8 | 很小 | 4:1 | 通用部署 |
| INT4 | 小 | 8:1 | 资源受限 |
| INT2 | 中等 | 16:1 | 极端压缩 |

#### 2. 剪枝 (Pruning)
```python
def prune_model(model, sparsity=0.5):
    # 移除不重要的权重连接
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=sparsity)
    return model
```

**方法分类**:
- **非结构化剪枝**: 移除个别权重
- **结构化剪枝**: 移除整个神经元/通道
- **动态剪枝**: 训练过程中自适应剪枝

#### 3. 知识蒸馏 (Knowledge Distillation)
```python
def distillation_loss(student_logits, teacher_logits, temperature=3.0):
    # 学生模型学习教师模型的"软"知识
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(soft_prob, soft_targets, reduction='batchmean')
```

**应用场景**:
- 🏫 大模型 → 小模型: GPT-3.5 → DistilBERT
- 🎯 任务专用: 通用模型 → 领域模型
- 📱 移动端部署: 服务器模型 → 手机模型

### 🚀 推理优化技术

#### 1. 算子融合 (Operator Fusion)
```python
# 将多个操作合并为单个内核
# 例如: LayerNorm + Linear → FusedLayerNormLinear
def fused_layer_norm_linear(x, weight, bias, norm_weight, norm_bias):
    # 一次性完成归一化和线性变换
    normalized = layer_norm(x, norm_weight, norm_bias)
    return F.linear(normalized, weight, bias)
```

#### 2. 动态批处理 (Dynamic Batching)
```python
class DynamicBatcher:
    def __init__(self, max_batch_size=32, max_wait_time=50):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
    
    async def add_request(self, request):
        self.pending_requests.append(request)
        if len(self.pending_requests) >= self.max_batch_size:
            return await self.process_batch()
        # 等待更多请求或超时
```

#### 3. KV缓存优化
```python
class KVCache:
    def __init__(self, max_seq_len, num_heads, head_dim):
        self.cache_k = torch.zeros(max_seq_len, num_heads, head_dim)
        self.cache_v = torch.zeros(max_seq_len, num_heads, head_dim)
        self.seq_len = 0
    
    def append(self, new_k, new_v):
        # 增量计算，避免重复计算历史token
        self.cache_k[self.seq_len] = new_k
        self.cache_v[self.seq_len] = new_v
        self.seq_len += 1
```

### 🏗️ 架构优化

#### 1. 混合精度训练 (Mixed Precision)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():  # FP16前向传播
        outputs = model(batch)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()  # 缩放梯度
    scaler.step(optimizer)
    scaler.update()
```

**优势**:
- 📊 内存占用减半
- ⚡ 训练速度提升 1.5-2x
- 🎯 保持数值稳定性

#### 2. 梯度检查点 (Gradient Checkpointing)
```python
def checkpoint_forward(function, *args):
    # 只保存部分激活值，需要时重新计算
    return torch.utils.checkpoint.checkpoint(function, *args)

# 在Transformer层中使用
class TransformerLayer(nn.Module):
    def forward(self, x):
        if self.training:
            return checkpoint_forward(self._forward_impl, x)
        else:
            return self._forward_impl(x)
```

#### 3. 并行策略

##### 数据并行 (Data Parallel)
```python
# 多GPU训练
model = nn.DataParallel(model)
# 或分布式数据并行
model = nn.parallel.DistributedDataParallel(model)
```

##### 模型并行 (Model Parallel)
```python
class ModelParallelTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # 不同层放在不同GPU上
        self.layer1 = TransformerLayer().to('cuda:0')
        self.layer2 = TransformerLayer().to('cuda:1')
    
    def forward(self, x):
        x = self.layer1(x.to('cuda:0'))
        x = self.layer2(x.to('cuda:1'))
        return x
```

##### 流水线并行 (Pipeline Parallel)
```python
# 使用FairScale或DeepSpeed
from fairscale.nn import Pipe

model = Pipe(transformer_layers, balance=[2, 2, 2, 2])
```

## 🚀 部署技术栈

### 1. 推理引擎对比

| 引擎 | 特点 | 适用场景 | 性能提升 |
|------|------|----------|----------|
| **ONNX Runtime** | 跨平台，易集成 | 通用部署 | 2-3x |
| **TensorRT** | NVIDIA优化 | GPU推理 | 3-5x |
| **OpenVINO** | Intel优化 | CPU推理 | 2-4x |
| **TVM** | 深度优化 | 定制硬件 | 3-6x |

### 2. 服务化部署

#### FastAPI + Uvicorn
```python
from fastapi import FastAPI
import torch

app = FastAPI()
model = load_optimized_model()

@app.post("/generate")
async def generate_text(prompt: str):
    with torch.no_grad():
        response = model.generate(prompt)
    return {"response": response}

# 启动: uvicorn main:app --workers 4
```

#### TorchServe
```bash
# 模型打包
torch-model-archiver --model-name transformer \
    --version 1.0 \
    --serialized-file model.pt \
    --handler custom_handler.py

# 部署服务
torchserve --start --model-store model_store \
    --models transformer=transformer.mar
```

### 3. 云端部署方案

#### Kubernetes + GPU
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-service
  template:
    metadata:
      labels:
        app: llm-service
    spec:
      containers:
      - name: llm-container
        image: llm-service:latest
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "16Gi"
            cpu: "4"
```

#### Auto Scaling
```python
# 基于队列长度的自动扩缩容
class AutoScaler:
    def __init__(self, min_replicas=1, max_replicas=10):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
    
    def should_scale_up(self, queue_length, avg_response_time):
        return queue_length > 50 or avg_response_time > 2000  # ms
    
    def should_scale_down(self, queue_length, avg_response_time):
        return queue_length < 10 and avg_response_time < 500  # ms
```

## 📊 性能监控

### 1. 关键指标
- **延迟**: P50, P95, P99响应时间
- **吞吐量**: QPS (Queries Per Second)
- **资源利用率**: GPU/CPU/内存使用率
- **准确率**: 模型输出质量监控

### 2. 监控实现
```python
import time
import psutil
import GPUtil

class PerformanceMonitor:
    def __init__(self):
        self.request_times = []
        self.gpu_usage = []
    
    def log_request(self, start_time, end_time):
        latency = (end_time - start_time) * 1000  # ms
        self.request_times.append(latency)
    
    def log_gpu_usage(self):
        gpus = GPUtil.getGPUs()
        if gpus:
            self.gpu_usage.append(gpus[0].load * 100)
    
    def get_metrics(self):
        return {
            'avg_latency': np.mean(self.request_times),
            'p95_latency': np.percentile(self.request_times, 95),
            'avg_gpu_usage': np.mean(self.gpu_usage)
        }
```

## 🔧 实践建议

### 1. 优化流程
```
基线测试 → 瓶颈分析 → 针对性优化 → 效果验证 → 持续监控
```

### 2. 优化优先级
1. **量化**: 最容易实现，效果显著
2. **算子融合**: 中等难度，稳定收益
3. **模型剪枝**: 需要重训练，长期收益
4. **知识蒸馏**: 高投入，高回报

### 3. 部署检查清单
- ✅ 模型格式转换 (PyTorch → ONNX/TensorRT)
- ✅ 推理引擎选择和配置
- ✅ 批处理策略设计
- ✅ 缓存机制实现
- ✅ 监控和告警设置
- ✅ 负载测试和容量规划

## 🌟 未来趋势

### 技术发展
- **硬件专用化**: AI芯片、神经网络处理器
- **软硬件协同**: 编译器优化、硬件感知训练
- **边缘部署**: 移动端、IoT设备优化
- **绿色AI**: 能耗优化、碳足迹减少

### 工程化成熟度
- **自动化工具**: 一键优化和部署
- **标准化流程**: MLOps最佳实践
- **生态完善**: 丰富的开源工具链

掌握这些优化和部署技术，是将AI研究成果转化为实际产品的关键！
