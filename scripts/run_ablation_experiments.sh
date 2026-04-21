#!/bin/bash
# Run all ablation experiments sequentially
# Usage: ./scripts/run_ablation_experiments.sh [--testing]
set -e

TESTING_FLAG=""
if [[ "${1}" == "--testing" ]]; then
    TESTING_FLAG="--testing"
fi

# MODELS=(ae_li ae_gaur ae_li_gaur_lex ae_li_gaur_synt ae_li_gaur_sem)
MODELS=(ae_li_gaur_synt ae_li_gaur_sem)
MODES=(lodo in_domain)

for model in "${MODELS[@]}"; do
    for mode in "${MODES[@]}"; do
        echo "========================================"
        echo "Running: $model -- $mode"
        echo "========================================"
        python3 scripts/submit_experiments.py \
            --model "$model" \
            --mode "$mode" \
            --no-matrix \
            --local \
            $TESTING_FLAG
    done
done

echo "All experiments done."
