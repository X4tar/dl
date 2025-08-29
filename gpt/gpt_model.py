"""
GPT (Generative Pre-trained Transformer) 完整实现
基于原始论文和现代最佳实践
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List

class GPTConfig:
    """GPT 模型配置"""
    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: Optional[int] = None,
        activation_function: str = "gelu",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        use_cache: bool = True,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner if n_inner is not None else 4 * n_embd
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache

class GPTAttention(nn.Module):
    """
    GPT 因果自注意力机制
    
    核心特点：
    1. 因果掩码确保只能看到当前位置之前的信息
    2. 多头并行计算
    3. 缩放点积注意力 + Dropout
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # Q, K, V 投影（合并计算提高效率）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # 因果掩码（下三角矩阵）
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                1, 1, config.n_positions, config.n_positions
            )
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, n_embd]
            attention_mask: 注意力掩码
            use_cache: 是否使用缓存（用于生成时的加速）
            past_key_value: 过去的 key 和 value
            
        Returns:
            output: 注意力输出 [batch_size, seq_len, n_embd]
            present_key_value: 当前的 key 和 value（如果use_cache=True）
        """
        B, T, C = x.size()  # batch_size, seq_len, n_embd
        
        # 计算 Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # 重塑为多头格式
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, n_head, T, head_dim]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # 如果使用缓存，连接过去的 key 和 value
        present_key_value = None
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)
        
        if use_cache:
            present_key_value = (k, v)
        
        # 计算注意力分数
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # 应用因果掩码
        seq_len_k = k.size(-2)
        causal_mask = self.bias[:, :, :T, :seq_len_k]
        att = att.masked_fill(causal_mask == 0, float('-inf'))
        
        # 应用额外的注意力掩码（如果提供）
        if attention_mask is not None:
            att = att + attention_mask
        
        # Softmax + Dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # 计算输出
        y = att @ v  # [B, n_head, T, head_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        
        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        
        return y, present_key_value

class GPTMLP(nn.Module):
    """
    GPT 前馈网络（MLP）
    
    结构：Linear -> GELU -> Linear -> Dropout
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.act = self._get_activation_function(config.activation_function)
        self.dropout = nn.Dropout(config.resid_pdrop)
        
    def _get_activation_function(self, name: str):
        """获取激活函数"""
        if name == "gelu":
            return F.gelu
        elif name == "relu":
            return F.relu
        elif name == "swish":
            return F.silu
        else:
            raise ValueError(f"Unsupported activation function: {name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class   GPTBlock(nn.Module):
    """
    GPT Transformer 块
    
    结构：
    1. Layer Norm + Multi-Head Attention + 残差连接
    2. Layer Norm + MLP + 残差连接
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPTAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPTMLP(config)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        # 自注意力
        residual = x
        x = self.ln_1(x)
        attn_output, present_key_value = self.attn(
            x, attention_mask=attention_mask, 
            use_cache=use_cache, past_key_value=past_key_value
        )
        x = residual + attn_output
        
        # 前馈网络
        residual = x
        x = self.ln_2(x)
        mlp_output = self.mlp(x)
        x = residual + mlp_output
        
        return x, present_key_value

class GPTModel(nn.Module):
    """
    GPT 主模型
    
    包含：
    1. 词嵌入 + 位置嵌入
    2. 多层 Transformer 块
    3. 最终层归一化
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # 词嵌入
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)  # 位置嵌入
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Transformer 块
        self.h = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)])
        
        # 最终层归一化
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor]]]]:
        """
        前向传播
        
        Args:
            input_ids: 输入token ids [batch_size, seq_len]
            attention_mask: 注意力掩码
            position_ids: 位置ids
            past_key_values: 过去的key-value缓存
            use_cache: 是否使用缓存
            
        Returns:
            last_hidden_state: 最后一层隐藏状态
            present_key_values: 当前的key-value缓存（如果use_cache=True）
        """
        batch_size, seq_len = input_ids.shape
        
        # 位置编码
        if position_ids is None:
            past_length = past_key_values[0][0].size(-2) if past_key_values is not None else 0
            position_ids = torch.arange(past_length, seq_len + past_length, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 词嵌入 + 位置嵌入
        token_embeddings = self.wte(input_ids)
        position_embeddings = self.wpe(position_ids)
        hidden_states = self.drop(token_embeddings + position_embeddings)
        
        # 处理注意力掩码
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # 通过所有 Transformer 块
        presents = [] if use_cache else None
        for i, block in enumerate(self.h):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_states, present = block(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_value=past_key_value
            )
            if use_cache:
                presents.append(present)
        
        # 最终层归一化
        hidden_states = self.ln_f(hidden_states)
        
        return hidden_states, presents

class GPTLMHeadModel(nn.Module):
    """
    GPT 语言建模头模型
    
    在 GPT 基础上添加语言建模头用于预测下一个词
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = GPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 权重共享（可选）
        self.lm_head.weight = self.transformer.wte.weight
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor]]]]:
        """
        前向传播
        
        Args:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            labels: 标签（用于计算损失）
            past_key_values: 过去的key-value缓存
            use_cache: 是否使用缓存
            
        Returns:
            logits: 输出logits
            loss: 损失（如果提供了labels）
            present_key_values: 当前的key-value缓存
        """
        # GPT 前向传播
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        
        hidden_states, present_key_values = transformer_outputs
        
        # 语言建模头
        logits = self.lm_head(hidden_states)
        
        # 计算损失
        loss = None
        if labels is not None:
            # 移位：预测下一个token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return logits, loss, present_key_values
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        文本生成
        
        Args:
            input_ids: 初始输入
            max_length: 最大生成长度
            temperature: 采样温度
            top_k: Top-K 采样
            top_p: Top-P (nucleus) 采样
            do_sample: 是否使用随机采样
            pad_token_id: padding token id
            eos_token_id: 结束token id
            
        Returns:
            generated_ids: 生成的token ids
        """
        self.eval()
        
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_length - input_ids.shape[1]):
            # 前向传播
            if past_key_values is None:
                model_inputs = {"input_ids": generated}
            else:
                model_inputs = {"input_ids": generated[:, -1:]}
            
            outputs = self.forward(
                **model_inputs,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            logits, _, past_key_values = outputs
            next_token_logits = logits[:, -1, :] / temperature
            
            # 采样策略
            if do_sample:
                if top_k is not None:
                    # Top-K 采样
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                if top_p is not None:
                    # Top-P (nucleus) 采样
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # 随机采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # 贪心解码
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 添加到生成序列
            generated = torch.cat([generated, next_token], dim=1)
            
            # 检查是否生成了结束符
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_gpt_model(model_size="small"):
    """创建不同规模的GPT模型"""
    if model_size == "nano":
        config = GPTConfig(
            vocab_size=50257,
            n_positions=256,
            n_embd=128,
            n_layer=4,
            n_head=4,
        )
    elif model_size == "small":
        config = GPTConfig(
            vocab_size=50257,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
        )
    elif model_size == "medium":
        config = GPTConfig(
            vocab_size=50257,
            n_positions=1024,
            n_embd=1024,
            n_layer=24,
            n_head=16,
        )
    elif model_size == "large":
        config = GPTConfig(
            vocab_size=50257,
            n_positions=1024,
            n_embd=1280,
            n_layer=36,
            n_head=20,
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    model = GPTLMHeadModel(config)
    print(f"Created GPT-{model_size} with {count_parameters(model):,} parameters")
    return model

if __name__ == "__main__":
    # 测试代码
    print("Testing GPT Model Implementation...")
    
    # 创建小型模型进行测试
    model = create_gpt_model("nano")
    
    # 创建测试输入
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    
    print(f"Input shape: {input_ids.shape}")
    
    # 前向传播测试
    with torch.no_grad():
        logits, loss, _ = model(input_ids, labels=input_ids)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Loss: {loss.item() if loss is not None else 'None'}")
    
    # 生成测试  
    print("\nTesting text generation...")
    prompt = torch.randint(0, 50257, (1, 10))
    generated = model.generate(prompt, max_length=20, temperature=0.8, do_sample=True)
    
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")
    
    print("\n✅ GPT model implementation test passed!")
