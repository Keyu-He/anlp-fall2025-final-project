# SAE Feature Steering 使用说明

## 简介

`qwen_sae_feature_steering.py` 脚本可以对 SAE 的特定 feature 进行 steering（引导/干预），观察模型输出的变化。

## 基本用法

```bash
python inferences/qwen_sae_feature_steering.py \
    --feature-idx <FEATURE_ID> \
    --steering-strength <STRENGTH> \
    --text "<YOUR_PROMPT>" \
    [其他参数...]
```

## 必需参数

- `--feature-idx`: 要干预的 feature 索引（从 CSV 中选择）
- `--text`: 输入的 prompt
- `--steering-strength`: 干预强度（默认 5.0）
  - 正值：增强该 feature
  - 负值：抑制该 feature

## 可选参数

- `--base-model-path`: 模型路径（默认：`/data/user_data/demiw/qwen2.5-7b-sotopia/Qwen2.5-7B-Instruct`）
- `--sae-dir`: SAE 目录（默认：`/data/user_data/demiw/qwen2.5-7b-sotopia/saes-qwen2.5-7b-instruct/resid_post_layer_15/trainer_1`）
- `--layer`: 层索引（默认：15）
- `--max-new-tokens`: 最大生成 token 数（默认：256）
- `--temperature`: 温度参数（默认：0.7）
- `--device`: 设备（默认：cuda）
- `--output-file`: 输出文件路径（可选，保存结果为 JSON）

## 示例

### 1. 基本使用 - 增强 relationship 相关特征

根据 `sae_top_features_for_steering.csv`，feature 325 与 relationship 正相关（0.5276）：

```bash
python inferences/qwen_sae_feature_steering.py \
    --feature-idx 325 \
    --steering-strength 5.0 \
    --text "Hello, I'm looking forward to working with you on this project."
```

### 2. 抑制 believability 相关特征

Feature 226 与 believability 负相关（-0.9182）：

```bash
python inferences/qwen_sae_feature_steering.py \
    --feature-idx 226 \
    --steering-strength -3.0 \
    --text "I need to convince you that this is the best solution."
```

### 3. 保存结果到文件

```bash
python inferences/qwen_sae_feature_steering.py \
    --feature-idx 116 \
    --steering-strength 3.0 \
    --text "Let's discuss the financial aspects of this deal." \
    --output-file results/steering_test.json
```

### 4. 使用 Sotopia prompt

```bash
python inferences/qwen_sae_feature_steering.py \
    --feature-idx 543 \
    --steering-strength 5.0 \
    --text "Imagine you are Alex Chen, a startup founder. You are negotiating with an investor. Turn #1: The investor asks about your business model. What do you say?" \
    --max-new-tokens 512
```

## Top Features 参考

根据 `sae_top_features_for_steering.csv`，以下是一些相关性较高的 features：

### Believability（负相关）
- 226, 485, 93, 37, 379 等（相关性 -0.9182）

### Financial & Material Benefits
- 15（正相关 0.5490）
- 401（负相关 -0.5347）

### Goal
- 86, 531, 196（负相关）

### Knowledge
- 545（正相关 0.4882）
- 388, 451, 78（正相关）

### Relationship
- 325（正相关 0.5276）
- 543, 113, 116（正相关）

## 输出说明

脚本会生成两个输出：

1. **Baseline Output**: 不进行任何 steering 的原始输出
2. **Steered Output**: 对指定 feature 进行 steering 后的输出

同时会显示：
- Baseline 时该 feature 的激活值
- 两个输出的对比

## 批量测试

使用提供的示例脚本：

```bash
bash inferences/run_steering_example.sh
```

或者自己编写循环测试多个 features：

```bash
# 测试不同 steering strengths
for strength in -5.0 -3.0 0.0 3.0 5.0; do
    python inferences/qwen_sae_feature_steering.py \
        --feature-idx 325 \
        --steering-strength $strength \
        --text "Your prompt here" \
        --output-file "results/steering_325_${strength}.json"
done
```

## 注意事项

1. **Feature 选择**：根据 CSV 文件中的相关性选择合适的 features
   - 正相关 feature：用正 strength 增强该维度
   - 负相关 feature：用负 strength 增强该维度

2. **Steering Strength**：
   - 建议从较小的值开始测试（如 ±3.0）
   - 过大的值可能导致输出不自然
   - 可以尝试 -5.0 到 5.0 范围内的值

3. **Layer 选择**：
   - 默认 layer 15（SAE 训练所在层）
   - 可以尝试其他层，但需要相应的 SAE

4. **Temperature**：
   - 0.0：确定性生成（便于对比）
   - 0.7：默认值，更自然
   - 1.0+：更随机，多样性更高
