# SAE Feature Steering 工具集

用于在 Sotopia benchmark 上进行 SAE feature steering 实验的工具集。

## 📁 目录结构

```
inferences/
├── README.md                          # 本文件
├── __init__.py                        # Python 包初始化
│
├── qwen_sae_feature_steering.py      # 基础 steering 脚本（支持自定义文本）
├── steering_with_sotopia.py          # Sotopia 任务 steering（从数据集加载）
├── batch_steering_test.py            # 批量测试脚本
│
├── explore_sotopia_data.py           # 浏览 Sotopia 数据集工具
├── run_batch_steering.sh             # 批量测试启动脚本
│
├── qwen_sae_inference_residual.py    # SAE inference 参考实现
├── qwen_sae_server_residual.py       # SAE server 参考实现
│
└── archive/                           # 已归档的旧文件
```

## 🚀 快速开始

### 1. 探索 Sotopia 数据集

查看所有可用的社交场景：

```bash
python inferences/explore_sotopia_data.py
```

输出包括：
- 90 个不同的社交场景
- 每个场景的参与者、描述、评分
- 第一轮对话示例

### 2. 单个 Feature Steering 测试

#### 方法 A：使用自定义文本

```bash
python inferences/qwen_sae_feature_steering.py \
    --feature-idx 325 \
    --steering-strength 1.0 \
    --text "Hello, let's work together on this project."
```

#### 方法 B：使用 Sotopia 任务

```bash
python inferences/steering_with_sotopia.py \
    --record-idx 0 \
    --feature-idx 325 \
    --steering-strength 1.0
```

### 3. 批量测试（推荐）

使用小强度（0.5-2.0）批量测试多个场景和 features：

```bash
python inferences/batch_steering_test.py --num-samples 16
```

或使用快捷脚本：

```bash
bash inferences/run_batch_steering.sh
```

## 📊 Top Features 参考

根据 `/home/demiw/anlp-fall2025-final-project/analysis/sae/sae_top_features_for_steering.csv`：

### Relationship（关系）
- **325** (相关性: 0.5276) - 最强关系特征
- **543** (相关性: 0.5122)
- **113** (相关性: 0.4920)
- **116** (相关性: 0.4539)

### Believability（可信度）
- **226, 485, 93, 37, 379, 315** 等（相关性: -0.9182，负相关）

### Knowledge（知识）
- **545** (相关性: 0.4882) - 最强知识特征
- **388** (相关性: 0.3893)
- **451** (相关性: 0.3829)
- **78** (相关性: 0.3609)

### Financial Benefits（财务利益）
- **15** (相关性: 0.5490) - 正相关
- **401** (相关性: -0.5347) - 负相关

### Goal（目标达成）
- **86** (相关性: -0.5070) - 负相关
- **531** (相关性: -0.4986)
- **196** (相关性: -0.4586)

## 🎯 推荐测试组合

### 测试 1: 议价场景中的关系维护

```bash
# Record 0 是家具议价场景
python inferences/steering_with_sotopia.py \
    --record-idx 0 \
    --feature-idx 325 \
    --steering-strength 1.0 \
    --output-file results/negotiation_relationship.json
```

**预期效果**: 更友好、更注重关系建立的回复

### 测试 2: 囚徒困境中的可信度

```bash
# Record 1 是囚徒困境场景
python inferences/steering_with_sotopia.py \
    --record-idx 1 \
    --feature-idx 226 \
    --steering-strength -1.0 \
    --output-file results/dilemma_believability.json
```

**预期效果**: 降低可信度，可能影响合作意愿

### 测试 3: 商业讨论中的专业知识

```bash
# Record 3 是商业伙伴讨论场景
python inferences/steering_with_sotopia.py \
    --record-idx 3 \
    --feature-idx 545 \
    --steering-strength 1.0 \
    --output-file results/business_knowledge.json
```

**预期效果**: 更专业、知识丰富的回答

### 测试 4: 不同强度对比

```bash
# 测试同一 feature 的不同强度效果
for strength in 0.5 1.0 1.5 2.0; do
    python inferences/steering_with_sotopia.py \
        --record-idx 0 \
        --feature-idx 325 \
        --steering-strength $strength \
        --output-file "results/strength_${strength}.json"
done
```

## 📝 参数说明

### 通用参数

- `--feature-idx`: Feature 索引（必需，从 CSV 中选择）
- `--steering-strength`: Steering 强度（推荐：0.5-2.0）
  - **正值**: 增强该 feature
  - **负值**: 抑制该 feature
  - **建议范围**: 0.5-2.0（避免过强导致重复输出）

### qwen_sae_feature_steering.py 特有参数

- `--text`: 输入文本（必需）
- `--sae-dir`: SAE 目录路径
- `--layer`: Layer 索引（默认: 15）
- `--max-new-tokens`: 最大生成 tokens（默认: 256）
- `--temperature`: 温度参数（默认: 0.7）
- `--output-file`: 保存结果到 JSON

### steering_with_sotopia.py 特有参数

- `--sotopia-data`: Sotopia 数据文件路径
- `--record-idx`: 记录索引（默认: 随机）
- `--turn-idx`: 对话轮次（默认: 1）
- 其他参数同上

### batch_steering_test.py 特有参数

- `--num-samples`: 测试样本数量（默认: 10）
- `--output-dir`: 输出目录（默认: results/batch_steering）

## 📂 Sotopia 数据集场景概览

数据集包含 90 个场景，类型包括：

| Record | 场景类型 | 描述 | 适合测试的维度 |
|--------|---------|------|---------------|
| 0 | 议价谈判 | 家具买卖 | relationship, financial, believability |
| 1 | 囚徒困境 | 两个罪犯的合作困境 | believability, goal |
| 2 | 社交偶遇 | 公园偶遇 | relationship |
| 3 | 商业讨论 | 财务状况评估 | knowledge, financial |
| 4 | 浪漫关系 | 情侣约会 | relationship |
| ... | ... | ... | ... |

使用 `explore_sotopia_data.py --show-all` 查看完整列表。

## 🔍 输出说明

所有脚本都会输出：

1. **Baseline Output**: 无 steering 的原始输出
2. **Steered Output**: 应用 steering 后的输出
3. **Baseline Feature Value**: 目标 feature 在 baseline 时的激活值
4. **对比摘要**: 两个输出的简短对比

### JSON 输出格式

使用 `--output-file` 保存的 JSON 包含：

```json
{
  "feature_idx": 325,
  "steering_strength": 1.0,
  "layer": 15,
  "prompt": "...",
  "baseline_output": "...",
  "steered_output": "...",
  "baseline_feature_value": 0.42,
  "config": {
    "max_new_tokens": 256,
    "temperature": 0.7
  }
}
```

## ⚠️ 注意事项

### 1. Steering 强度选择

- **过小**（< 0.5）: 效果可能不明显
- **适中**（0.5-2.0）: **推荐范围**，效果明显且输出自然
- **过大**（> 3.0）: 可能导致输出重复或不自然

### 2. Feature 选择建议

- **正相关 feature**: 用正强度增强该维度
- **负相关 feature**: 用负强度增强该维度（或正强度抑制）
- 优先测试相关性较高的 features（|correlation| > 0.4）

### 3. 温度参数

- `temperature=0.0`: 确定性生成，便于对比分析
- `temperature=0.7`: 默认值，输出更自然但有随机性
- `temperature=1.0+`: 更多样但可能不稳定

### 4. 输出重复问题

如果遇到输出重复：
- ✅ 降低 steering strength（使用 0.5-1.5）
- ✅ 减少 max_new_tokens（使用 64-128）
- ✅ 调整 temperature（尝试 0.8-1.0）

## 📈 分析建议

### 1. 强度-效果关系

测试不同强度的效果曲线：

```bash
for strength in 0.0 0.5 1.0 1.5 2.0 2.5 3.0; do
    python inferences/steering_with_sotopia.py \
        --record-idx 0 --feature-idx 325 --steering-strength $strength \
        --output-file "results/strength_analysis_${strength}.json"
done
```

### 2. 跨场景一致性

测试同一 feature 在不同场景的效果：

```bash
for record in 0 1 2 3 4 5; do
    python inferences/steering_with_sotopia.py \
        --record-idx $record --feature-idx 325 --steering-strength 1.0 \
        --output-file "results/cross_scenario_record${record}.json"
done
```

### 3. Feature 对比

对比不同 relationship features：

```bash
for feature in 325 543 113 116; do
    python inferences/steering_with_sotopia.py \
        --record-idx 0 --feature-idx $feature --steering-strength 1.0 \
        --output-file "results/feature_comparison_f${feature}.json"
done
```

### 4. 批量测试与评分

使用批量测试脚本，然后用 Sotopia 评分模型评估 steered outputs 的分数变化：

```bash
python inferences/batch_steering_test.py --num-samples 20
# 然后使用评分模型评估 results/batch_steering/batch_results_*.jsonl
```

## 🛠️ 技术细节

### SAE Steering 原理

1. **Forward Hook**: 在指定 layer 的 MLP 输出处拦截
2. **Encode**: 使用 SAE encoder 将 residual stream 编码为 sparse features
3. **Intervene**: 修改目标 feature 的激活值（加上 steering_strength）
4. **Decode**: 使用 SAE decoder 解码回 residual stream
5. **Replace**: 用修改后的 residual 替换原始 MLP 输出

### 模型与 SAE 配置

- **模型**: Qwen2.5-7B-Instruct
- **SAE Layer**: 15 (resid_post_layer_15)
- **SAE Type**: Top-K SAE (k=64)
- **Feature 数量**: 约 576 个

### Dtype 处理

代码自动处理 dtype 转换：
- 模型使用 `bfloat16`
- SAE 使用 `float32`
- Steering 后自动转回 `bfloat16`

## 📚 参考文件

- **Top Features CSV**: `/home/demiw/anlp-fall2025-final-project/analysis/sae/sae_top_features_for_steering.csv`
- **Sotopia 数据**: `/home/demiw/anlp-fall2025-final-project/results/sotopia_all_gpt-4o_Qwen_Qwen2.5-7B-Instruct_20251201_merged.jsonl`
- **SAE 模型**: `/data/user_data/demiw/qwen2.5-7b-sotopia/saes-qwen2.5-7b-instruct/resid_post_layer_15/trainer_1`

## 🐛 常见问题

### Q: 输出一直重复怎么办？
A: 降低 steering_strength 到 0.5-1.5 范围，或减少 max_new_tokens。

### Q: 如何判断 steering 是否有效？
A: 对比 baseline 和 steered 输出的语气、态度、内容重点。可以使用 Sotopia 评分模型量化评估。

### Q: 哪些 features 效果最明显？
A: 通常高相关性的 features（|corr| > 0.5）效果更明显，如 feature 325 (relationship)、226 (believability)。

### Q: 可以同时 steer 多个 features 吗？
A: 当前脚本只支持单 feature steering。如需多 feature，需修改 `attach_steering_hook` 函数。

### Q: 为什么有些场景效果不明显？
A: Feature 的效果可能依赖场景。某些 features 在特定类型的对话中更活跃。

## 📞 需要帮助？

如有问题，请查看：
1. `explore_sotopia_data.py` - 了解数据集
2. `archive/` 目录下的详细文档
3. 代码中的注释和 docstrings

---

**最后更新**: 2025-12-03
**作者**: SAE Steering Project
**项目**: Sotopia Benchmark SAE Interpretability Analysis
