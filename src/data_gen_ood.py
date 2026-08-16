"""Generate an out-of-distribution (OOD) test set: grids much bigger than the
training distribution (configs/default.yaml trains on grid_size_range
[6, 12]), to test which approach -- the fine-tuned LLM or the PPO agent --
generalizes better to a scale/shape neither ever saw during training.

Why this matters: PPO's observation encodes the goal direction/distance
normalized by a fixed constant (max_size_for_norm, matched to training's
grid_size_range upper bound) -- feed it a much bigger grid and those numbers
fall outside what it was ever trained on, which is exactly the failure mode
fixed-size neural policies are prone to. The LLM sees the grid as plain
text, so nothing in its architecture hard-codes a maximum grid size; whether
that flexibility translates into better OOD generalization (rather than just
"different failure mode") is the empirical question this test set is for.

Usage:
    python -m src.data_gen_ood --grid_size_range 18 26 --n 100 \
        --out data/test_ood.jsonl --seed 4242
"""
import argparse
import random

from src.data_gen import generate_split, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size_range", type=int, nargs=2, default=[18, 26])
    parser.add_argument("--obstacle_density_range", type=float, nargs=2, default=[0.10, 0.25])
    parser.add_argument("--min_optimal_len", type=int, default=3)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--out", default="data/test_ood.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    examples = generate_split(
        args.n, args.grid_size_range, args.obstacle_density_range,
        args.min_optimal_len, rng,
    )
    write_jsonl(examples, args.out)
    lengths = [e["meta"]["optimal_len"] for e in examples]
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    sizes = [e["meta"]["instance"]["size"] for e in examples]
    print(f"Wrote {len(examples)} OOD examples to {args.out} "
          f"(grid sizes {min(sizes)}-{max(sizes)}, avg optimal path length = {avg_len:.1f})")


if __name__ == "__main__":
    main()
