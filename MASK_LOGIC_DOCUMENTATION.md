# Mask逻辑梳理文档

## 1. 数据准备阶段 (rl_dataset.py)

### 1.1 Prompt构建
- **格式**: `PROMPT2 + "\n" + text + "\n" + "<|endoftext|>"`
  - `PROMPT2`: "Read and analyze the following text, then you need to provide your reasoning within <think></think> tags. Finally, generate a comprehensive understanding of this text."
  - `text`: query/positive/negative的实际内容
  - `<|endoftext|>`: 在qwen3中是pad_token

### 1.2 Tokenization
- 使用 `tokenizer.encode(prompt, add_special_tokens=False)` 得到 `input_ids`
- 初始 `attention_mask` 全为1（所有token都参与计算）

### 1.3 截断逻辑 (truncate_with_eot_preserved)
- 如果序列长度 > max_prompt_length，进行截断
- **关键**: 只截断中间内容部分（text），保留：
  - 前缀：PROMPT2 + "\n"
  - 后缀：<|endoftext|>
- 截断后重新生成 `attention_mask`，所有token（包括<|endoftext|>）都设为1

### 1.4 Padding (postprocess_data)
- 使用 `left_pad=True`，padding在左侧
- 有效token在右侧
- padding位置的 `attention_mask` 设为0

### 1.5 最终数据结构
- `input_ids`: [batch_size, max_prompt_length]
- `attention_mask`: [batch_size, max_prompt_length]
  - 左侧padding: 0
  - 有效token（包括PROMPT2、text、<|endoftext|>）: 1
  - 右侧padding（如果有）: 0

## 2. 模型前向传播

### 2.1 输入
- `input_ids`: token IDs
- `attention_mask`: 用于attention计算，padding位置为0

### 2.2 输出
- `hidden_states`: [batch_size, seq_len, hidden_dim]
  - 包含所有位置的hidden states，包括system prompt、content和<|endoftext|>

## 3. Grace.py中的Pooling逻辑

### 3.1 输入
- `hidden_states`: [batch_size, seq_len, hidden_dim] - 模型输出的hidden states
- `attention_masks`: [batch_size, seq_len] - 来自数据的attention_mask

### 3.2 Pooling Mask生成 (extract_mean_pooled_hidden_states_batch)

#### 步骤1: 初始化pooling_mask
```python
pooling_mask = attention_masks_float.clone()  # 初始化为attention_mask的副本
```

#### 步骤2: System Prompt Exclusion (如果启用)
如果 `exclude_system_prompt=True`:
1. 获取system prompt长度: `system_prompt_len = _get_system_prompt_token_length()`
   - 计算PROMPT2 + "\n"的token长度
2. 对于每个样本:
   - 找到有效token的起始位置（第一个attention_mask=1的位置）
   - 计算system prompt结束位置: `system_prompt_end_pos = valid_start_pos + system_prompt_len`
   - 将system prompt部分mask掉: `pooling_mask[k, valid_start_pos:system_prompt_end_pos] = 0`

#### 步骤3: 应用Mask
```python
pooling_mask_expanded = pooling_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
masked_hidden_states = hidden_states * pooling_mask_expanded  # 被mask的位置变为0
```

#### 步骤4: Mean Pooling
```python
valid_lengths = pooling_mask.sum(dim=1, keepdim=True)  # 统计有效token数量
sum_hidden_states = masked_hidden_states.sum(dim=1)  # 对seq_len维度求和
mean_pooled_hidden_states = sum_hidden_states / valid_lengths  # 平均
```

#### 步骤5: L2 Normalization
```python
mean_pooled_hidden_states = F.normalize(mean_pooled_hidden_states, p=2, dim=1)
```

### 3.3 最终输出
- `mean_pooled_hidden_states`: [batch_size, hidden_dim]
  - 用于计算相似度和reward

## 4. Mask逻辑总结

### 4.1 哪些部分会被mask？

1. **Padding位置** (attention_mask=0)
   - 位置: 左侧padding（left_pad=True）
   - 原因: 不是真实token，只是填充

2. **System Prompt部分** (如果exclude_system_prompt=True)
   - 位置: PROMPT2 + "\n"部分
   - 原因: 排除system prompt对embedding的影响
   - 实现: 在pooling_mask中将对应位置设为0

3. **不被mask的部分**:
   - Content部分（text）
   - <|endoftext|> token（在qwen3中是pad_token，但在attention_mask中设为1）

### 4.2 用来计算embedding的tensor

1. **输入tensor**:
   - `hidden_states`: 模型输出的所有位置的hidden states
   - `attention_masks`: 来自数据的attention_mask（标识padding）

2. **中间tensor**:
   - `pooling_mask`: 在attention_mask基础上，进一步mask掉system prompt
   - `masked_hidden_states`: hidden_states * pooling_mask_expanded

3. **输出tensor**:
   - `mean_pooled_hidden_states`: 经过mean pooling和L2 normalization的embedding

### 4.3 如何产生？

1. **hidden_states**: 模型前向传播产生
2. **attention_masks**: 数据准备阶段产生（rl_dataset.py）
3. **pooling_mask**: 在extract_mean_pooled_hidden_states_batch中基于attention_masks和system_prompt_len产生
4. **mean_pooled_hidden_states**: 通过masked_hidden_states的mean pooling产生

## 5. 修改后的影响

### 5.1 System Prompt Exclusion仍然生效
- ✅ `_get_system_prompt_token_length()`已更新以适应新的next token prediction格式
- ✅ 计算PROMPT2 + "\n"的token长度
- ✅ 在pooling时正确排除system prompt部分

### 5.2 需要注意的点
1. **<|endoftext|>的处理**:
   - 在qwen3中是pad_token，但在attention_mask中设为1
   - 会参与pooling计算（这是期望的行为）

2. **System Prompt长度计算**:
   - 使用新的格式: `PROMPT2 + "\n" + text + "\n" + "<|endoftext|>"`
   - 通过查找user content的起始位置来确定system prompt长度

3. **Padding位置**:
   - 使用left_pad=True，padding在左侧
   - 有效token在右侧，pooling_mask会正确识别
