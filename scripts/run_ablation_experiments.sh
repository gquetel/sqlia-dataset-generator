#!/bin/bash
# Run all ablation experiments sequentially 
set -e

# MODELS=(ae_li ae_gaur ae_li_gaur_lex ae_li_gaur_synt ae_li_gaur_sem)
MODELS=(ae_li_gaur_lex ae_li_gaur_synt ae_li_gaur_sem)
MODES=(generic specialised)

for model in "${MODELS[@]}"; do
    for mode in "${MODES[@]}"; do
        echo "========================================"
        echo "Running: $model -- $mode"
        echo "========================================"
        python3 scripts/submit_experiments.py \
            --model "$model" \
            --mode "$mode" \
            --no-matrix \
            --local
    done
done

echo "All experiments done."
