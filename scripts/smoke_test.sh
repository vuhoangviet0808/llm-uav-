#!/usr/bin/env bash
# End-to-end pipeline sanity check: tiny model, tiny data, CPU, no internet
# needed. Run this first, on any machine, before touching the real 1B model.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/4 generating tiny dataset =="
python -m src.data_gen --config configs/smoke_test.yaml

echo "== 2/4 fine-tuning tiny debug model (LoRA) =="
python -m src.train --config configs/smoke_test.yaml

echo "== 3/4 evaluating fine-tuned tiny model =="
python -m src.evaluate --config configs/smoke_test.yaml --adapter_dir outputs_smoke/lora-tiny \
    --report_path outputs_smoke/eval_report_finetuned.json

echo "== 4/4 evaluating base (not fine-tuned) tiny model for comparison =="
python -m src.evaluate --config configs/smoke_test.yaml \
    --report_path outputs_smoke/eval_report_base.json

echo "Done. Compare outputs_smoke/eval_report_base.json vs outputs_smoke/eval_report_finetuned.json"
