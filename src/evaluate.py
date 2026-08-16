"""Evaluate a (fine-tuned or base) model on the pathfinding test set.

Reports, in the same spirit as the RPD metric in the "Learning to Recharge"
paper (agent steps vs. heuristic/optimal steps):
  - success_rate: fraction of episodes where the model's moves reach G
                  without ever crossing an obstacle / leaving the grid
  - avg_length_ratio: (model path length) / (BFS-optimal length), averaged
                       over successful episodes only (1.0 = optimal)
  - invalid_move_rate: fraction of episodes where the model tried an illegal
                        move before reaching the goal
  - unparseable_rate: fraction of episodes where no valid move token could be
                       parsed out of the model's raw output at all

Usage:
    python -m src.evaluate --config configs/default.yaml \
        [--adapter_dir outputs/lora-pathfinding] [--limit 50]
"""
import argparse
import json
import os

import torch
import yaml
from tqdm import tqdm

from src.env import GridInstance, apply_moves
from src.model import load_base_model
from src.prompts import parse_moves
from src.tokenizer import get_tokenizer


def load_model_for_eval(model_name, adapter_dir, tiny_vocab_size=None):
    if adapter_dir and model_name == "tiny-debug":
        # train.py full-fine-tunes the tiny debug model (see comment there),
        # so this is a plain checkpoint, not a LoRA adapter.
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(adapter_dir)
    else:
        model = load_base_model(model_name, tiny_vocab_size=tiny_vocab_size)
        if adapter_dir:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_dir)
            model = model.merge_and_unload()
    model.eval()
    return model


@torch.no_grad()
def generate_moves(model, tokenizer, prompt: str, max_new_tokens: int = 64):
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    pad_id = tokenizer.pad_token_id
    eos_id = getattr(tokenizer, "eos_token_id", None)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
    )
    new_tokens = out[0][input_ids.shape[1]:].tolist()
    return tokenizer.decode(new_tokens)


def evaluate(model, tokenizer, test_examples, max_new_tokens=64):
    n = len(test_examples)
    n_success = 0
    n_invalid = 0
    n_unparseable = 0
    length_ratios = []

    per_example = []
    # Each iteration calls model.generate() once (no batching), which can take
    # several seconds per example on a free-tier GPU -- a bare `for` loop here
    # prints nothing until the very end, which is easy to mistake for a hang.
    # tqdm gives a live per-example progress bar + running success rate.
    progress = tqdm(test_examples, desc="evaluating", unit="ex")
    for ex in progress:
        inst = GridInstance.from_dict(ex["meta"]["instance"])
        optimal_len = ex["meta"]["optimal_len"]

        raw = generate_moves(model, tokenizer, ex["prompt"], max_new_tokens)
        moves = parse_moves(raw)

        if len(moves) == 0 and optimal_len > 0:
            n_unparseable += 1

        result = apply_moves(inst, moves)
        success = result.reached_goal
        invalid = result.hit_invalid_at is not None

        if success:
            n_success += 1
            # length actually used to reach the goal = number of steps taken
            used_len = len(result.visited) - 1
            if optimal_len > 0:
                length_ratios.append(used_len / optimal_len)
            else:
                length_ratios.append(1.0)
        if invalid:
            n_invalid += 1

        done = len(per_example) + 1
        progress.set_postfix(success=f"{n_success}/{done}", invalid=n_invalid)

        per_example.append({
            "prompt_size": inst.size,
            "optimal_len": optimal_len,
            "model_raw_output": raw,
            "model_moves": moves,
            "success": success,
            "invalid_move": invalid,
        })

    report = {
        "n_examples": n,
        "success_rate": n_success / n if n else 0.0,
        "invalid_move_rate": n_invalid / n if n else 0.0,
        "unparseable_rate": n_unparseable / n if n else 0.0,
        "avg_length_ratio_on_success": (sum(length_ratios) / len(length_ratios)
                                        if length_ratios else None),
    }
    return report, per_example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--adapter_dir", default=None,
                         help="Path to a saved LoRA adapter. Omit to evaluate the base model zero-shot.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only evaluate the first N test examples (useful for a quick check).")
    parser.add_argument("--report_path", default=None,
                         help="Override cfg.eval.report_path (handy to keep base-vs-fine-tuned reports separate).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    tokenizer = get_tokenizer(model_name)
    tiny_vocab_size = tokenizer.vocab_size if model_name == "tiny-debug" else None

    model = load_model_for_eval(model_name, args.adapter_dir, tiny_vocab_size)

    test_path = os.path.join(cfg["data"]["out_dir"], "test.jsonl")
    with open(test_path) as f:
        test_examples = [json.loads(line) for line in f]
    if args.limit:
        test_examples = test_examples[:args.limit]

    report, per_example = evaluate(model, tokenizer, test_examples,
                                    max_new_tokens=cfg["eval"]["max_new_tokens"])

    print(json.dumps(report, indent=2))

    report_path = args.report_path or cfg["eval"]["report_path"]
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({"summary": report, "examples": per_example}, f, indent=2)
    print(f"Full report written to {report_path}")


if __name__ == "__main__":
    main()
