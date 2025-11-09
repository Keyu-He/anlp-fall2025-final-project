# anlp-fall2025-final-project

## Sotopia Baseline Benchmark

This repository implements baseline benchmarking for the Sotopia social intelligence evaluation framework.

### Overview

The benchmark evaluates social intelligence in language agents by running simulations on the Sotopia Hard tasks (100 scenarios). The baseline includes two models:

- **gpt-4o** (OpenAI)
- **DeepSeek-R1-Distill-Llama-8B** (DeepSeek)

### Prerequisites

1. Install `uv` package manager:
   ```bash
   pip install uv
   ```

2. Set up the Sotopia environment:
   ```bash
   cd sotopia
   uv sync --all-extras
   ```

3. Set up required API keys:
   - **OpenAI API Key** (for gpt-4o): Set the `OPENAI_API_KEY` environment variable
   - **DeepSeek API Key** (for DeepSeek-R1-Distill-Llama-8B): Set the `DEEPSEEK_API_KEY` environment variable

   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export DEEPSEEK_API_KEY="your-deepseek-key"
   ```

### Running the Benchmark

#### Run benchmarks for both baseline models:
```bash
python run_baseline_benchmark.py
```

#### Display existing results only (no new runs):
```bash
python run_baseline_benchmark.py --only-show-performance
```

#### Run benchmarks for specific models:
```bash
python run_baseline_benchmark.py --models gpt-4o --models deepseek/deepseek-r1-distill-llama-8b
```

### Benchmark Details

- **Task**: Sotopia Hard tasks (100 scenarios)
- **Partner Model**: `meta-llama/Llama-3-70b-chat-hf` (fixed)
- **Evaluator Model**: `gpt-4o` (default)
- **Evaluation Dimensions**:
  - Social Rules (SOC)
  - Secret (SEC)
  - Financial and Material Benefits (FIN)
  - Relationship (REL)
  - Knowledge (KNO)
  - Goal (GOAL)
  - Believability (BEL)

### Output

The benchmark will generate:
1. Console output showing progress and results
2. Episode logs stored in the database
3. Performance metrics with confidence intervals

### Additional Resources

- [Sotopia Documentation](https://docs.sotopia.world)
- [Sotopia Paper](https://arxiv.org/abs/2310.11667)
- [Sotopia GitHub](https://github.com/sotopia-lab/sotopia)