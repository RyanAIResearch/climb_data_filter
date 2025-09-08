#!/usr/bin/env bash
set -euo pipefail

LMFLOW_DIR=../LMFlow                # Your LMFlow root directory (data4elm branch)
DATA_DIR=$(pwd)/data/filtered_textonly
OUT_DIR=output_models/dora
MERGED=output_models/dora_merged

# 1) Single-round DoRA (keep official recipe)
bash $LMFLOW_DIR/scripts/train.sh \
  --dataset_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR" \

# 2) Merge DoRA weights
bash $LMFLOW_DIR/scripts/run_merge_dora.sh \
  --model_name_or_path data4elm/Llama-400M-12L \
  --lora_model_path "$OUT_DIR" \
  --output_model_path "$MERGED"

# 3) Evaluate four ELMB tasks
cd $LMFLOW_DIR/lm-evaluation-harness
lm_eval --model hf \
  --model_args pretrained=$(realpath $MERGED),trust_remote_code=True \
  --tasks elmb_roleplay,elmb_reasoning,elmb_functioncalling,elmb_chatrag \
  --device cuda:0 --batch_size 1 --log_samples \
  --output_path ./eval_results/test_elmb

# Print results
echo "============================================"
echo "ELMB Evaluation Results:"
cat ./eval_results/test_elmb/results.json | python -m json.tool