#!/usr/bin/env python3
"""
Run Sotopia baseline benchmarks for specified models.

This script runs the Sotopia benchmark on two baseline models:
- gpt-4o (OpenAI)
- DeepSeek-R1-Distill-Llama-8B (DeepSeek)

The benchmark evaluates social intelligence in language agents by running
simulations on the Sotopia Hard tasks (100 scenarios).

Usage:
    python run_baseline_benchmark.py [--only-show-performance]
    
    --only-show-performance: Only display results without running new benchmarks
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_benchmark(models: list[str], only_show_performance: bool = False) -> None:
    """
    Run the Sotopia benchmark for specified models.
    
    Args:
        models: List of model names to benchmark
        only_show_performance: If True, only show existing results
    """
    # Navigate to sotopia directory
    sotopia_dir = Path(__file__).parent / "sotopia"
    
    # Build the command
    cmd = ["uv", "run", "sotopia", "benchmark"]
    
    # Add models to benchmark
    for model in models:
        cmd.extend(["--models", model])
    
    # Add optional flags
    if only_show_performance:
        cmd.append("--only-show-performance")
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Working directory: {sotopia_dir}")
    print("-" * 80)
    
    # Run the benchmark
    try:
        result = subprocess.run(
            cmd,
            cwd=sotopia_dir,
            check=True,
            capture_output=False,
            text=True
        )
        print("-" * 80)
        print("✓ Benchmark completed successfully")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running benchmark: {e}", file=sys.stderr)
        return e.returncode
    except KeyboardInterrupt:
        print("\n✗ Benchmark interrupted by user", file=sys.stderr)
        return 130


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Run Sotopia baseline benchmarks for gpt-4o and DeepSeek-R1-Distill-Llama-8B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--only-show-performance",
        action="store_true",
        help="Only display results without running new benchmarks"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o", "deepseek/deepseek-r1-distill-llama-8b"],
        help=(
            "Models to benchmark (default: gpt-4o and deepseek-r1-distill-llama-8b). "
            "Note: DeepSeek-R1-Distill-Llama-8B may also be available as: "
            "together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-8B or "
            "openrouter/deepseek/deepseek-r1-distill-llama-8b"
        )
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Sotopia Baseline Benchmark")
    print("=" * 80)
    print(f"Models to benchmark: {', '.join(args.models)}")
    print(f"Mode: {'Display only' if args.only_show_performance else 'Run + Display'}")
    print("=" * 80)
    print()
    
    return run_benchmark(args.models, args.only_show_performance)


if __name__ == "__main__":
    sys.exit(main())
