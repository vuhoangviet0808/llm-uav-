"""Generate train/val/test JSONL datasets of (prompt, completion) pairs for
SFT fine-tuning + evaluation, using BFS as the ground-truth planner.

Usage:
    python -m src.data_gen --config configs/default.yaml
"""
import argparse
import json
import os
import random

import yaml

from src.env import random_instance
from src.prompts import build_completion, build_prompt


def generate_split(n: int, grid_size_range, obstacle_density_range, min_optimal_len, rng):
    examples = []
    for _ in range(n):
        inst, path = random_instance(
            size_range=tuple(grid_size_range),
            obstacle_density=tuple(obstacle_density_range),
            min_optimal_len=min_optimal_len,
            rng=rng,
        )
        examples.append({
            "prompt": build_prompt(inst),
            "completion": build_completion(path),
            "meta": {
                "instance": inst.to_dict(),
                "optimal_moves": path,
                "optimal_len": len(path),
            },
        })
    return examples


def write_jsonl(examples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg["data"]

    rng = random.Random(dcfg["seed"])
    out_dir = dcfg["out_dir"]

    splits = {
        "train": dcfg["train_size"],
        "val": dcfg["val_size"],
        "test": dcfg["test_size"],
    }
    for name, n in splits.items():
        examples = generate_split(
            n,
            dcfg["grid_size_range"],
            dcfg["obstacle_density_range"],
            dcfg.get("min_optimal_len", 3),
            rng,
        )
        path = os.path.join(out_dir, f"{name}.jsonl")
        write_jsonl(examples, path)
        lengths = [e["meta"]["optimal_len"] for e in examples]
        avg_len = sum(lengths) / len(lengths) if lengths else 0.0
        print(f"[{name}] wrote {len(examples)} examples to {path} "
              f"(avg optimal path length = {avg_len:.1f})")


if __name__ == "__main__":
    main()
