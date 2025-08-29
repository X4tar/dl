"""
完整的 BERT 模型实现
包括预训练任务（MLM, NSP）和下游任务适配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from bert_components import BERTEmbeddings, BERTEncoder, BERTPooler

class BERTModel(nn.Module):
    """
    BERT 基础模型
    包含嵌入层、编码器和池化层
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 核心组件
        self.embeddings = BERTEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=config.type_vocab_size,
            dropout=config.hidden_dropout_prob
        )
        
        self.encoder = BERTEncoder(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            dropout=config.hidden_dropout_prob
        )
        
        self.pooler = BERTPooler(config.hidden_size)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, return_attention=False):
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len] 输入词汇 ID
            attention_mask: [batch_size, seq_len] 注意力掩码
            token_type_ids: [batch_size, seq_len] 句子类型 ID
            position_ids: [batch_size, seq_len] 位置 ID
            return_attention: 是否返回注意力权重
            
        Returns:
            sequence_output: [batch_size, seq_len, hidden_size] 序列表示
            pooled_output: [batch_size, hidden_size] 池化表示
            attention_weights: 注意力权重（可选）
        """
        # 获取输入嵌入
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )
        
        # 通过编码器
        if return_attention:
            sequence_output, attention_weights = self.encoder(
                embedding_output, 
                attention_mask=attention_mask,
                return_attention=True
            )
        else:
            sequence_output = self.encoder(
                embedding_output,
                attention_mask=attention_mask
            )
            attention_weights = None
        
        # 池化
        pooled_output = self.pooler(sequence_output)
        
        if return_attention:
            return sequence_output, pooled_output, attention_weights
        else:
            return sequence_output, pooled_output


class BERTForMaskedLM(nn.Module):
    """
    用于掩码语言模型（MLM）预训练的 BERT
    """
    
    def __init__(self, config):
        super().__init__()
        self.bert = BERTModel(config)
        
        # MLM 预测头
        self.cls = BERTMLMHead(config)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                labels=None, return_attention=False):
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len] 输入词汇 ID
            attention_mask: [batch_size, seq_len] 注意力掩码
            token_type_ids: [batch_size, seq_len] 句子类型 ID
            labels: [batch_size, seq_len] MLM 标签
            return_attention: 是否返回注意力权重
            
        Returns:
            loss: MLM 损失（如果提供 labels）
            prediction_scores: [batch_size, seq_len, vocab_size] 预测分数
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_attention=return_attention
        )
        
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        
        outputs = (prediction_scores,) + outputs[1:]
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.bert.config.vocab_size),
                labels.view(-1)
            )
            outputs = (masked_lm_loss,) + outputs
        
        return outputs


class BERTForNextSentencePrediction(nn.Module):
    """
    用于下一句预测（NSP）预训练的 BERT
    """
    
    def __init__(self, config):
        super().__init__()
        self.bert = BERTModel(config)
        
        # NSP 预测头
        self.cls = BERTNSPHead(config)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                labels=None, return_attention=False):
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len] 输入词汇 ID
            attention_mask: [batch_size, seq_len] 注意力掩码
            token_type_ids: [batch_size, seq_len] 句子类型 ID
            labels: [batch_size] NSP 标签
            return_attention: 是否返回注意力权重
            
        Returns:
            loss: NSP 损失（如果提供 labels）
            prediction_scores: [batch_size, 2] 预测分数
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_attention=return_attention
        )
        
        pooled_output = outputs[1]
        prediction_scores = self.cls(pooled_output)
        
        outputs = (prediction_scores,) + outputs[2:]
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            next_sentence_loss = loss_fct(prediction_scores, labels)
            outputs = (next_sentence_loss,) + outputs
        
        return outputs


class BERTForPreTraining(nn.Module):
    """
    用于 BERT 预训练的完整模型
    同时包含 MLM 和 NSP 任务
    """
    
    def __init__(self, config):
        super().__init__()
        self.bert = BERTModel(config)
        
        # 预训练头
        self.cls = BERTPreTrainingHeads(config)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                masked_lm_labels=None, next_sentence_label=None, return_attention=False):
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len] 输入词汇 ID
            attention_mask: [batch_size, seq_len] 注意力掩码
            token_type_ids: [batch_size, seq_len] 句子类型 ID
            masked_lm_labels: [batch_size, seq_len] MLM 标签
            next_sentence_label: [batch_size] NSP 标签
            return_attention: 是否返回注意力权重
            
        Returns:
            total_loss: 总损失
            prediction_scores: MLM 预测分数
            seq_relationship_scores: NSP 预测分数
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_attention=return_attention
        )
        
        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        
        outputs = (prediction_scores, seq_relationship_score) + outputs[2:]
        
        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            
            # MLM 损失
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.bert.config.vocab_size),
                masked_lm_labels.view(-1)
            )
            
            # NSP 损失
            next_sentence_loss = loss_fct(seq_relationship_score, next_sentence_label)
            
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs
        
        return outputs


class BERTForSequenceClassification(nn.Module):
    """
    用于序列分类的 BERT
    适用于情感分析、文本分类等任务
    """
    
    def __init__(self, config, num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BERTModel(config)
        
        # 分类头
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                labels=None, return_attention=False):
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len] 输入词汇 ID
            attention_mask: [batch_size, seq_len] 注意力掩码
            token_type_ids: [batch_size, seq_len] 句子类型 ID
            labels: [batch_size] 分类标签
            return_attention: 是否返回注意力权重
            
        Returns:
            loss: 分类损失（如果提供 labels）
            logits: [batch_size, num_labels] 分类 logits
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_attention=return_attention
        )
        
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs = (logits,) + outputs[2:]
        
        if labels is not None:
            if self.num_labels == 1:
                # 回归任务
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # 分类任务
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs
        
        return outputs


class BERTForTokenClassification(nn.Module):
    """
    用于词级分类的 BERT
    适用于命名实体识别、词性标注等任务
    """
    
    def __init__(self, config, num_labels=9):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BERTModel(config)
        
        # 词级分类头
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                labels=None, return_attention=False):
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len] 输入词汇 ID
            attention_mask: [batch_size, seq_len] 注意力掩码
            token_type_ids: [batch_size, seq_len] 句子类型 ID
            labels: [batch_size, seq_len] 词级标签
            return_attention: 是否返回注意力权重
            
        Returns:
            loss: 分类损失（如果提供 labels）
            logits: [batch_size, seq_len, num_labels] 词级 logits
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_attention=return_attention
        )
        
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        outputs = (logits,) + outputs[2:]
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # 只对有效位置（非填充）计算损失
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        
        return outputs


class BERTForQuestionAnswering(nn.Module):
    """
    用于问答任务的 BERT
    预测答案的起始和结束位置
    """
    
    def __init__(self, config):
        super().__init__()
        self.bert = BERTModel(config)
        
        # 问答头
        self.qa_outputs = nn.Linear(config.hidden_size, 2)  # start 和 end
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                start_positions=None, end_positions=None, return_attention=False):
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len] 输入词汇 ID
            attention_mask: [batch_size, seq_len] 注意力掩码
            token_type_ids: [batch_size, seq_len] 句子类型 ID
            start_positions: [batch_size] 答案开始位置
            end_positions: [batch_size] 答案结束位置
            return_attention: 是否返回注意力权重
            
        Returns:
            loss: 问答损失（如果提供位置标签）
            start_logits: [batch_size, seq_len] 开始位置 logits
            end_logits: [batch_size, seq_len] 结束位置 logits
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_attention=return_attention
        )
        
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        outputs = (start_logits, end_logits) + outputs[2:]
        
        if start_positions is not None and end_positions is not None:
            # 计算起始和结束位置的损失
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            
            # 忽略超出序列长度的位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        
        return outputs


# 预训练头部类定义

class BERTMLMHead(nn.Module):
    """BERT MLM 预测头"""
    
    def __init__(self, config):
        super().__init__()
        self.transform = BERTLMPredictionHead(config)
        
        # 输出层权重与词嵌入权重绑定
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias
    
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BERTNSPHead(nn.Module):
    """BERT NSP 预测头"""
    
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
    
    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BERTPreTrainingHeads(nn.Module):
    """BERT 预训练头部集合"""
    
    def __init__(self, config):
        super().__init__()
        self.predictions = BERTMLMHead(config)
        self.seq_relationship = BERTNSPHead(config)
    
    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BERTLMPredictionHead(nn.Module):
    """BERT LM 预测变换头"""
    
    def __init__(self, config):
        super().__init__()
        self.transform = BERTPredictionHeadTransform(config)
    
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        return hidden_states


class BERTPredictionHeadTransform(nn.Module):
    """BERT 预测头变换层"""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = F.gelu
        self.layer_norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


# 配置类

class BERTConfig:
    """BERT 模型配置"""
    
    def __init__(self, 
                 vocab_size=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


def test_bert_models():
    """测试各种 BERT 模型"""
    print("=" * 60)
    print("测试 BERT 模型变体")
    print("=" * 60)
    
    # 创建小型配置用于测试
    config = BERTConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=128
    )
    
    batch_size = 2
    seq_len = 20
    
    # 创建测试数据
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    
    print(f"配置参数:")
    print(f"  词汇表大小: {config.vocab_size}")
    print(f"  隐藏层大小: {config.hidden_size}")
    print(f"  层数: {config.num_hidden_layers}")
    print(f"  注意力头数: {config.num_attention_heads}")
    
    # 1. 测试基础 BERT 模型
    print("\n1. 测试基础 BERT 模型")
    print("-" * 30)
    
    bert_model = BERTModel(config)
    sequence_output, pooled_output = bert_model(input_ids, attention_mask, token_type_ids)
    
    print(f"序列输出形状: {sequence_output.shape}")
    print(f"池化输出形状: {pooled_output.shape}")
    
    # 2. 测试序列分类模型
    print("\n2. 测试序列分类模型")
    print("-" * 30)
    
    classification_model = BERTForSequenceClassification(config, num_labels=3)
    labels = torch.randint(0, 3, (batch_size,))
    
    outputs = classification_model(input_ids, attention_mask, token_type_ids, labels)
    loss, logits = outputs[0], outputs[1]
    
    print(f"分类损失: {loss.item():.4f}")
    print(f"分类 logits 形状: {logits.shape}")
    print(f"预测类别: {torch.argmax(logits, dim=-1)}")
    
    # 3. 测试词级分类模型
    print("\n3. 测试词级分类模型")
    print("-" * 30)
    
    token_classification_model = BERTForTokenClassification(config, num_labels=5)
    token_labels = torch.randint(0, 5, (batch_size, seq_len))
    
    outputs = token_classification_model(input_ids, attention_mask, token_type_ids, token_labels)
    loss, logits = outputs[0], outputs[1]
    
    print(f"词级分类损失: {loss.item():.4f}")
    print(f"词级 logits 形状: {logits.shape}")
    
    # 4. 测试问答模型
    print("\n4. 测试问答模型")
    print("-" * 30)
    
    qa_model = BERTForQuestionAnswering(config)
    start_positions = torch.randint(0, seq_len, (batch_size,))
    end_positions = torch.randint(0, seq_len, (batch_size,))
    
    outputs = qa_model(input_ids, attention_mask, token_type_ids, 
                      start_positions, end_positions)
    loss, start_logits, end_logits = outputs[0], outputs[1], outputs[2]
    
    print(f"问答损失: {loss.item():.4f}")
    print(f"开始位置 logits 形状: {start_logits.shape}")
    print(f"结束位置 logits 形状: {end_logits.shape}")
    
    # 5. 参数统计
    print("\n5. 模型参数统计")
    print("-" * 30)
    
    models = [
        ("基础模型", bert_model),
        ("序列分类", classification_model),
        ("词级分类", token_classification_model),
        ("问答模型", qa_model)
    ]
    
    for name, model in models:
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {params:,} 参数")
    
    print("\n✅ BERT 模型测试完成!")
    return True


if __name__ == "__main__":
    test_bert_models()
    
    print("\n" + "=" * 60)
    print("BERT 模型模块测试完成！")
    print("=" * 60)
