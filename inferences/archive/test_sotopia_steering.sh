#!/bin/bash

# Test feature steering with real Sotopia tasks
# This script tests different features on various Sotopia scenarios

SOTOPIA_DATA="/home/demiw/anlp-fall2025-final-project/results/sotopia_all_gpt-4o_Qwen_Qwen2.5-7B-Instruct_20251201_merged.jsonl"

echo "=========================================="
echo "Testing Feature Steering with Sotopia Tasks"
echo "=========================================="
echo ""

# Test 1: Relationship feature on negotiation task (record 0 - furniture negotiation)
echo "Test 1: Enhance relationship feature (325) on negotiation task"
echo "Expected: More cooperative, relationship-building response"
echo "----------------------------------------"
python inferences/steering_with_sotopia.py \
    --sotopia-data "$SOTOPIA_DATA" \
    --record-idx 0 \
    --turn-idx 1 \
    --feature-idx 325 \
    --steering-strength 5.0 \
    --max-new-tokens 256 \
    --output-file "results/sotopia_steering_negotiation_relationship.json"

echo ""
echo ""

# Test 2: Believability feature on prisoner's dilemma (record 1)
echo "Test 2: Suppress believability feature (226) on prisoner's dilemma"
echo "Expected: Less believable/trustworthy response"
echo "----------------------------------------"
python inferences/steering_with_sotopia.py \
    --sotopia-data "$SOTOPIA_DATA" \
    --record-idx 1 \
    --turn-idx 1 \
    --feature-idx 226 \
    --steering-strength -3.0 \
    --max-new-tokens 256 \
    --output-file "results/sotopia_steering_dilemma_believability.json"

echo ""
echo ""

# Test 3: Knowledge feature on business partners (record 3)
echo "Test 3: Enhance knowledge feature (545) on business discussion"
echo "Expected: More knowledgeable, informative response"
echo "----------------------------------------"
python inferences/steering_with_sotopia.py \
    --sotopia-data "$SOTOPIA_DATA" \
    --record-idx 3 \
    --turn-idx 1 \
    --feature-idx 545 \
    --steering-strength 5.0 \
    --max-new-tokens 256 \
    --output-file "results/sotopia_steering_business_knowledge.json"

echo ""
echo ""

# Test 4: Financial benefits feature on negotiation
echo "Test 4: Enhance financial benefits feature (15) on negotiation"
echo "Expected: More financially focused response"
echo "----------------------------------------"
python inferences/steering_with_sotopia.py \
    --sotopia-data "$SOTOPIA_DATA" \
    --record-idx 0 \
    --turn-idx 1 \
    --feature-idx 15 \
    --steering-strength 5.0 \
    --max-new-tokens 256 \
    --output-file "results/sotopia_steering_negotiation_financial.json"

echo ""
echo "=========================================="
echo "All tests completed!"
echo "Check results/ directory for output files"
echo "=========================================="
