#!/usr/bin/env python3
"""
Test script to verify the Sotopia baseline benchmark setup.

This script checks that:
1. Required dependencies are installed
2. Environment variables are set
3. Models can be accessed
4. Sotopia benchmark command is available
"""

import os
import sys
import subprocess
from pathlib import Path


def check_dependency(module_name: str, import_name: str = None) -> bool:
    """Check if a Python module is installed."""
    if import_name is None:
        import_name = module_name
    
    try:
        __import__(import_name)
        print(f"✓ {module_name} is installed")
        return True
    except ImportError:
        print(f"✗ {module_name} is NOT installed")
        return False


def check_env_var(var_name: str, required: bool = True) -> bool:
    """Check if an environment variable is set."""
    value = os.environ.get(var_name)
    if value:
        masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
        print(f"✓ {var_name} is set: {masked}")
        return True
    else:
        status = "✗" if required else "⚠"
        msg = "NOT set (REQUIRED)" if required else "NOT set (optional)"
        print(f"{status} {var_name} is {msg}")
        return not required


def check_uv() -> bool:
    """Check if uv is installed."""
    try:
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ uv is installed: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ uv is NOT installed")
        return False


def check_sotopia_cli() -> bool:
    """Check if sotopia CLI is available."""
    sotopia_dir = Path(__file__).parent / "sotopia"
    try:
        result = subprocess.run(
            ["uv", "run", "sotopia", "--help"],
            cwd=sotopia_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        print("✓ Sotopia CLI is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print("✗ Sotopia CLI is NOT available")
        return False


def main():
    """Run all checks."""
    print("=" * 80)
    print("Sotopia Baseline Benchmark - Setup Verification")
    print("=" * 80)
    print()
    
    all_checks_passed = True
    
    # Check dependencies
    print("Checking dependencies...")
    print("-" * 80)
    all_checks_passed &= check_uv()
    print()
    
    # Check Python packages (in sotopia environment)
    print("Checking Python packages (will be checked by sotopia CLI)...")
    print("-" * 80)
    print("Note: These will be installed when you run 'cd sotopia && uv sync --all-extras'")
    print()
    
    # Check environment variables
    print("Checking environment variables...")
    print("-" * 80)
    all_checks_passed &= check_env_var("OPENAI_API_KEY", required=True)
    all_checks_passed &= check_env_var("DEEPSEEK_API_KEY", required=False)
    all_checks_passed &= check_env_var("TOGETHER_API_KEY", required=False)
    all_checks_passed &= check_env_var("OPENROUTER_API_KEY", required=False)
    print()
    
    # Check Sotopia CLI
    print("Checking Sotopia CLI...")
    print("-" * 80)
    all_checks_passed &= check_sotopia_cli()
    print()
    
    # Summary
    print("=" * 80)
    if all_checks_passed:
        print("✓ All checks passed! You're ready to run the benchmark.")
        print()
        print("To run the benchmark:")
        print("  python run_baseline_benchmark.py")
        print()
        print("To only display existing results:")
        print("  python run_baseline_benchmark.py --only-show-performance")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above before running the benchmark.")
        print()
        print("Setup instructions:")
        print("1. Install uv: pip install uv")
        print("2. Install dependencies: cd sotopia && uv sync --all-extras")
        print("3. Set API keys:")
        print("   export OPENAI_API_KEY='your-key'")
        print("   export DEEPSEEK_API_KEY='your-key' (optional)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
