# Sotopia Feature Steering 使用指南

## 概述

这套工具可以让你使用真实的 Sotopia 任务数据来测试 SAE feature steering，观察不同 features 对模型行为的影响。

## 文件说明

1. **steering_with_sotopia.py** - 主脚本：从 Sotopia 数据中加载任务并进行 steering
2. **explore_sotopia_data.py** - 数据探索工具：查看所有 Sotopia 场景
3. **test_sotopia_steering.sh** - 批量测试脚本：测试多个场景和 features

## 快速开始

### 1. 探索可用的 Sotopia 场景

```bash
python inferences/explore_sotopia_data.py
```

这会显示前 20 个场景，包括：
- 场景描述
- 参与者（agents）
- 评分（believability, relationship, knowledge, goal, financial）
- 第一轮对话

### 2. 选择一个场景测试 steering

```bash
# 基本用法：测试 record 0（家具议价），增强 relationship feature
python inferences/steering_with_sotopia.py \
    --record-idx 0 \
    --feature-idx 325 \
    --steering-strength 5.0
```

### 3. 运行预设的批量测试

```bash
bash inferences/test_sotopia_steering.sh
```

这会自动测试 4 个场景和不同的 features 组合。

## 详细使用

### steering_with_sotopia.py 参数

**必需参数：**
- `--feature-idx`: 要 steer 的 feature 索引

**常用参数：**
- `--record-idx`: Sotopia 记录索引（默认：随机）
- `--turn-idx`: 使用第几轮对话（默认：1）
- `--steering-strength`: Steering 强度（默认：5.0）
- `--max-new-tokens`: 最大生成 tokens（默认：256）
- `--temperature`: 温度参数（默认：0.7）
- `--output-file`: 保存结果到 JSON 文件

**示例：**

```bash
# 测试 record 0，增强 relationship
python inferences/steering_with_sotopia.py \
    --record-idx 0 \
    --feature-idx 325 \
    --steering-strength 5.0 \
    --output-file results/test.json

# 测试 record 1，抑制 believability
python inferences/steering_with_sotopia.py \
    --record-idx 1 \
    --feature-idx 226 \
    --steering-strength -3.0

# 随机选择一个场景测试
python inferences/steering_with_sotopia.py \
    --feature-idx 545 \
    --steering-strength 3.0
```

## 推荐测试组合

根据你的 CSV 文件（`sae_top_features_for_steering.csv`），以下是一些有趣的测试组合：

### 1. 议价场景（Record 0）- 家具买卖

**测试 relationship features:**
```bash
# 增强 relationship (325, correlation: 0.5276)
python inferences/steering_with_sotopia.py --record-idx 0 --feature-idx 325 --steering-strength 5.0
python inferences/steering_with_sotopia.py --record-idx 0 --feature-idx 543 --steering-strength 5.0

# 预期：更注重关系维护，可能更愿意妥协
```

**测试 financial benefits features:**
```bash
# 增强 financial focus (15, correlation: 0.5490)
python inferences/steering_with_sotopia.py --record-idx 0 --feature-idx 15 --steering-strength 5.0

# 预期：更关注经济利益，可能更强硬
```

**测试 believability features:**
```bash
# 抑制 believability (226, correlation: -0.9182)
python inferences/steering_with_sotopia.py --record-idx 0 --feature-idx 226 --steering-strength -3.0

# 预期：可能不太真实或可信
```

### 2. 囚徒困境场景（Record 1）- 两个罪犯

**测试 believability:**
```bash
# 增强/抑制可信度
python inferences/steering_with_sotopia.py --record-idx 1 --feature-idx 226 --steering-strength 5.0
python inferences/steering_with_sotopia.py --record-idx 1 --feature-idx 226 --steering-strength -5.0

# 预期：影响合作意愿和承诺可信度
```

**测试 goal features:**
```bash
# 测试 goal-related features
python inferences/steering_with_sotopia.py --record-idx 1 --feature-idx 86 --steering-strength -5.0

# 预期：影响达成目标的意愿
```

### 3. 商业伙伴场景（Record 3）- 财务讨论

**测试 knowledge features:**
```bash
# 增强 knowledge (545, correlation: 0.4882)
python inferences/steering_with_sotopia.py --record-idx 3 --feature-idx 545 --steering-strength 5.0
python inferences/steering_with_sotopia.py --record-idx 3 --feature-idx 388 --steering-strength 5.0

# 预期：更专业、知识丰富的回答
```

**测试 financial features:**
```bash
python inferences/steering_with_sotopia.py --record-idx 3 --feature-idx 15 --steering-strength 5.0

# 预期：更关注财务细节
```

### 4. 公园偶遇场景（Record 2）- 社交互动

**测试 relationship features:**
```bash
python inferences/steering_with_sotopia.py --record-idx 2 --feature-idx 325 --steering-strength 5.0

# 预期：更友好、建立联系的态度
```

## Sotopia 数据集概览

数据集包含 90 个不同的社交场景，类型包括：
- 议价/谈判（negotiation）
- 囚徒困境（prisoner's dilemma）
- 商业讨论（business discussions）
- 浪漫关系（romantic relationships）
- 随机偶遇（casual encounters）
- 等等...

每个场景包含：
- 详细的情境描述
- 两个角色的背景信息和目标
- 多轮对话记录
- 各维度的评分（believability, relationship, knowledge, goal, financial, social_rules, secret）

## 分析建议

### 1. 对比分析

保存结果到 JSON 文件，然后对比分析：

```bash
# 测试不同 strengths
for strength in -5.0 -3.0 0.0 3.0 5.0; do
    python inferences/steering_with_sotopia.py \
        --record-idx 0 \
        --feature-idx 325 \
        --steering-strength $strength \
        --output-file "results/record0_f325_s${strength}.json"
done
```

### 2. 多场景测试

对同一个 feature 在不同场景下测试：

```bash
# 测试 relationship feature 在不同场景
for record in 0 1 2 3 4; do
    python inferences/steering_with_sotopia.py \
        --record-idx $record \
        --feature-idx 325 \
        --steering-strength 5.0 \
        --output-file "results/f325_record${record}.json"
done
```

### 3. Feature 对比

在同一场景测试不同 features：

```bash
# Record 0 测试不同的 relationship-related features
for feature in 325 543 113 116; do
    python inferences/steering_with_sotopia.py \
        --record-idx 0 \
        --feature-idx $feature \
        --steering-strength 5.0 \
        --output-file "results/record0_f${feature}.json"
done
```

## 预期发现

根据你的分析目标，可以观察：

1. **Feature 与 Dimension 的关系**
   - 高相关性的 features 是否真的改变了对应维度的行为？
   - 例如：steering relationship features 是否让回复更友好合作？

2. **Steering Strength 的影响**
   - 不同强度的效果差异
   - 是否存在最优强度？
   - 过强的 steering 是否产生不自然的输出？

3. **场景依赖性**
   - 同一 feature 在不同场景下的效果是否一致？
   - 某些场景是否对 steering 更敏感？

4. **评分变化**
   - 如果你有评分模型，可以测试 steered output 的各维度分数是否按预期变化

## 输出格式

脚本会输出：
1. **场景信息**：scenario, agents, record/turn index
2. **Steering 配置**：feature index, strength, layer
3. **Baseline output**：无 steering 的原始输出
4. **Steered output**：steering 后的输出
5. **Feature activation**：该 feature 在 baseline 时的激活值
6. **Summary**：两个输出的对比摘要

如果指定 `--output-file`，会保存完整的 JSON 结果，包含所有元数据。

## 常见场景类型

在数据集中寻找特定类型的场景：

```bash
# 查看所有场景（包含评分）
python inferences/explore_sotopia_data.py --show-all

# 找 relationship 分数高的场景
python inferences/explore_sotopia_data.py --show-all | grep -A 10 "Relationship: [89]"

# 找商业/财务相关场景
python inferences/explore_sotopia_data.py --show-all | grep -i "business\|financial"
```

## 注意事项

1. **温度参数**：
   - temperature=0.0 可以得到确定性输出，便于对比
   - temperature=0.7 更自然但有随机性

2. **Turn 选择**：
   - turn_idx=1 通常是第一轮回复（推荐）
   - 可以测试后续 turns，但上下文会更复杂

3. **结果解读**：
   - 观察输出的语气、态度、内容重点
   - 不仅看"说了什么"，也看"怎么说的"
   - 注意是否产生了不自然或不连贯的文本

4. **Baseline 重要性**：
   - 总是先看 baseline 输出
   - 只有与 baseline 对比才能看出 steering 效果
