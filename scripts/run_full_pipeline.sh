#!/usr/bin/env bash
# Real run: fine-tune a ~1B-parameter open LLM with LoRA on the pathfinding
# task. Needs a GPU + internet access to huggingface.co (and, for the
# default Llama-3.2-1B-Instruct, a HF token with the license accepted:
# https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct -> "Agree and access")
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:-configs/default.yaml}

echo "== 1/4 generating dataset =="
python -m src.data_gen --config "$CONFIG"

echo "== 2/4 evaluating the BASE model zero-shot (before fine-tuning) =="
python -m src.evaluate --config "$CONFIG" --report_path outputs/eval_report_base.json --limit 50

echo "== 3/4 LoRA fine-tuning =="
python -m src.train --config "$CONFIG"

echo "== 4/4 evaluating the FINE-TUNED model =="
OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['train']['output_dir'])")
python -m src.evaluate --config "$CONFIG" --adapter_dir "$OUTPUT_DIR" \
    --report_path outputs/eval_report_finetuned.json

echo "Done. Compare outputs/eval_report_base.json vs outputs/eval_report_finetuned.json"
