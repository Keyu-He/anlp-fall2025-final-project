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

   You can either export them directly:
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export DEEPSEEK_API_KEY="your-deepseek-key"
   ```
   
   Or create a `.env` file from the example:
   ```bash
   cp .env.example .env
   # Then edit .env and add your API keys
   ```

### Running the Benchmark

#### Option 1: Using the shell script (recommended)
```bash
./run_benchmark.sh
```

This script will automatically load environment variables from `.env` file if it exists.

#### Option 2: Using the Python script directly
```bash
python run_baseline_benchmark.py
```

#### Display existing results only (no new runs):
```bash
python run_baseline_benchmark.py --only-show-performance
# or
./run_benchmark.sh --only-show-performance
```

#### Run benchmarks for specific models:
```bash
python run_baseline_benchmark.py --models gpt-4o --models deepseek/deepseek-r1-distill-llama-8b
```

### Setup Verification

Before running the benchmark, you can verify your setup:
```bash
python test_setup.py
```

This will check:
- Required dependencies are installed
- Environment variables are set correctly
- Sotopia CLI is available

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

## Repository Structure

```
.
├── README.md                    # This file
├── run_baseline_benchmark.py    # Main Python script to run benchmarks
├── run_benchmark.sh             # Convenience shell wrapper script
├── test_setup.py                # Setup verification script
├── .env.example                 # Example environment configuration
└── sotopia/                     # Sotopia submodule
```

### File Descriptions

- **run_baseline_benchmark.py**: Main Python script that wraps the Sotopia benchmark command. Supports flexible model selection and output options.
- **run_benchmark.sh**: Shell wrapper that automatically loads `.env` file and runs the Python script.
- **test_setup.py**: Verification script to check if all dependencies and API keys are properly configured.
- **.env.example**: Template for environment configuration. Copy to `.env` and fill in your API keys.