#!/bin/bash
# Convenience wrapper script for running Sotopia baseline benchmarks

set -e

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env file exists and load it
if [ -f "$SCRIPT_DIR/.env" ]; then
    print_info "Loading environment variables from .env file..."
    export $(cat "$SCRIPT_DIR/.env" | grep -v '^#' | xargs)
else
    print_warning ".env file not found. Using existing environment variables."
    print_info "To create a .env file, copy .env.example and fill in your API keys:"
    print_info "  cp .env.example .env"
fi

# Check if required API keys are set
if [ -z "$OPENAI_API_KEY" ]; then
    print_error "OPENAI_API_KEY is not set!"
    print_info "Please set it in .env file or export it:"
    print_info "  export OPENAI_API_KEY='your-key'"
    exit 1
fi

# Run the Python script with all arguments passed through
print_info "Running Sotopia baseline benchmark..."
python3 "$SCRIPT_DIR/run_baseline_benchmark.py" "$@"
