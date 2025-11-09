# Sotopia Baseline Benchmark - Technical Details

## Overview

This document provides technical details about the Sotopia baseline benchmark implementation.

## What the Benchmark Does

The Sotopia benchmark evaluates social intelligence in language agents through simulated social interactions. The benchmark:

1. **Loads Scenarios**: Uses predefined social scenarios from the Sotopia Hard task set (100 scenarios)
2. **Creates Agents**: Instantiates two agents - the test model and a partner model
3. **Runs Simulations**: Executes conversations between agents in various social scenarios
4. **Evaluates Performance**: Uses an evaluator model to score the interactions across multiple dimensions

## Evaluation Dimensions

Each interaction is evaluated on 7 dimensions:

| Dimension | Abbreviation | Range | Description |
|-----------|-------------|-------|-------------|
| Social Rules | SOC | [-10, 0] | Adherence to social norms and rules |
| Secret | SEC | [-10, 0] | Ability to keep secrets appropriately |
| Financial and Material Benefits | FIN | [-5, 5] | Success in financial/material goals |
| Relationship | REL | [-5, 5] | Quality of relationship building |
| Knowledge | KNO | [0, 10] | Information exchange and learning |
| Goal | GOAL | [0, 10] | Achievement of conversation goals |
| Believability | BEL | [0, 10] | Naturalness and believability of responses |

## Model Configuration

### Test Models (Baseline)
- **gpt-4o**: OpenAI's GPT-4 Optimized model
- **DeepSeek-R1-Distill-Llama-8B**: DeepSeek's reasoning model distilled to Llama architecture

### Partner Model
- **together_ai/meta-llama/Llama-3-70b-chat-hf**: Fixed partner model for all benchmarks
  - Ensures consistent comparison across test models
  - Hosted on Together AI platform

### Evaluator Model
- **gpt-4o**: Used to evaluate conversation quality
  - Provides scores for each dimension
  - Generates reasoning for scores

## Model Name Formats

Different providers require different model name formats:

### OpenAI Models
```
gpt-4o
gpt-4o-mini
gpt-3.5-turbo
```

### DeepSeek Models
Via DeepSeek API:
```
deepseek/deepseek-chat
deepseek/deepseek-r1-distill-llama-8b
```

Via Together AI:
```
together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

Via OpenRouter:
```
openrouter/deepseek/deepseek-r1-distill-llama-8b
```

### Together AI Models
```
together_ai/meta-llama/Llama-3-70b-chat-hf
together_ai/mistralai/Mixtral-8x22B-Instruct-v0.1
together_ai/meta-llama/Llama-3-8b-chat-hf
```

## Benchmark Process

1. **Initialization**
   - Load environment configurations
   - Initialize database connections (Redis)
   - Fetch benchmark scenarios

2. **Episode Generation**
   - For each scenario:
     - Create environment with scenario details
     - Instantiate agents with test and partner models
     - Run conversation (max 20 turns)
     - Evaluate conversation quality

3. **Evaluation**
   - Evaluator model scores each dimension
   - Scores are averaged across all episodes
   - Confidence intervals calculated (95% CI)

4. **Results**
   - Display performance metrics
   - Save results to database
   - Optionally export to JSONL format

## Performance Metrics

Results include:
- **Mean Score**: Average score across all episodes
- **Confidence Interval**: 95% CI showing statistical uncertainty
- **Setting Count**: Number of unique scenarios tested
- **Episode Count**: Total number of episodes run

## Expected Runtime

- **Per Episode**: ~30-60 seconds (depending on model speed)
- **Full Benchmark (100 scenarios)**: ~1-2 hours
- **Factors Affecting Speed**:
  - Model response time
  - API rate limits
  - Network latency
  - Batch size (default: 10)

## API Requirements

### Required APIs
- **OpenAI API**: For gpt-4o model and evaluator
  - Set: `OPENAI_API_KEY`

### Optional APIs (for DeepSeek)
Choose one of:
- **DeepSeek API**: Direct access to DeepSeek models
  - Set: `DEEPSEEK_API_KEY`
- **Together AI**: Access to DeepSeek and other models
  - Set: `TOGETHER_API_KEY`
- **OpenRouter**: Unified access to multiple providers
  - Set: `OPENROUTER_API_KEY`

## Troubleshooting

### Redis Connection Error
```
Failed to connect to Redis: Error 111 connecting to localhost:6379
```
**Solution**: This is a warning and can be safely ignored for display-only mode (`--only-show-performance`). For running new benchmarks, set up Redis:
```bash
docker run -d -p 6379:6379 redis:latest
```

### Model Not Found
```
Error: Model 'deepseek/...' not found
```
**Solution**: Try alternative model name formats:
- `deepseek/deepseek-r1-distill-llama-8b`
- `together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- `openrouter/deepseek/deepseek-r1-distill-llama-8b`

### API Rate Limits
**Solution**: Reduce batch size:
```bash
python run_baseline_benchmark.py --batch-size 5
```

Or use the sotopia CLI directly with batch size:
```bash
cd sotopia
uv run sotopia benchmark --models gpt-4o --batch-size 5
```

## Output Format

### Console Output
```
Model: gpt-4o, episodes: 200, Avg Rewards: {...}
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ Model      ┃ believability┃ Settings ┃ Episodes ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ gpt-4o     │ 7.00 ± 3.45  │ 100      │ 200      │
└────────────┴──────────────┴──────────┴──────────┘
```

### JSONL Output (if enabled)
```json
{"model_name": "gpt-4o", "SOC [-10, 0]": 7.0, "SEC [-10, 0]": 7.0, ...}
```

## References

- [Sotopia Paper](https://arxiv.org/abs/2310.11667)
- [Sotopia Documentation](https://docs.sotopia.world)
- [LiteLLM Documentation](https://docs.litellm.ai/)
