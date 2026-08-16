"""Evaluate a trained PPO agent on the pathfinding test set, using the exact
same metrics/schema as src.evaluate so results are directly comparable with
the LLM (zero-shot and fine-tuned) numbers.

`unparseable_rate` doesn't apply to PPO (it always emits one of the 4
discrete actions, never malformed output) and is reported as 0.0 for schema
parity with the LLM's eval_report_*.json files.

Usage:
    python -m src.evaluate_rl --config configs/default.yaml \
        --model_path outputs/ppo-pathfinding/ppo_model.zip
"""
import argparse
import json
import os

import yaml
from stable_baselines3 import PPO
from tqdm import tqdm

from src.env import GridInstance
from src.rl_env import ACTIONS, PathfindingEnv


def evaluate(model, test_examples, dcfg):
    n = len(test_examples)
    n_success = 0
    n_invalid = 0
    length_ratios = []
    per_example = []

    env = PathfindingEnv(
        grid_size_range=dcfg["grid_size_range"],
        obstacle_density_range=dcfg["obstacle_density_range"],
        max_size_for_norm=dcfg["grid_size_range"][1],
    )

    progress = tqdm(test_examples, desc="evaluating PPO", unit="ex")
    for ex in progress:
        inst = GridInstance.from_dict(ex["meta"]["instance"])
        optimal_len = ex["meta"]["optimal_len"]

        env.inst = inst
        env.pos = inst.start
        env.steps = 0
        env.max_episode_steps = env.steps_per_size * inst.size
        obs = env._encode()

        moves = []
        success = False
        invalid = False
        for _ in range(env.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            moves.append(ACTIONS[int(action)])
            if info.get("invalid"):
                invalid = True
                break
            if info.get("success"):
                success = True
                break
            if truncated:
                break

        if success:
            n_success += 1
            used_len = len(moves)  # every move up to and including the goal-reaching one
            length_ratios.append(used_len / optimal_len if optimal_len > 0 else 1.0)
        if invalid:
            n_invalid += 1

        done = len(per_example) + 1
        progress.set_postfix(success=f"{n_success}/{done}", invalid=n_invalid)

        per_example.append({
            "prompt_size": inst.size,
            "optimal_len": optimal_len,
            "model_moves": moves,
            "success": success,
            "invalid_move": invalid,
        })

    report = {
        "n_examples": n,
        "success_rate": n_success / n if n else 0.0,
        "invalid_move_rate": n_invalid / n if n else 0.0,
        "unparseable_rate": 0.0,
        "avg_length_ratio_on_success": (sum(length_ratios) / len(length_ratios)
                                        if length_ratios else None),
    }
    return report, per_example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model_path", default="outputs/ppo-pathfinding/ppo_model.zip")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report_path", default="outputs/eval_report_ppo.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg["data"]

    model = PPO.load(args.model_path)

    test_path = os.path.join(dcfg["out_dir"], "test.jsonl")
    with open(test_path) as f:
        test_examples = [json.loads(line) for line in f]
    if args.limit:
        test_examples = test_examples[:args.limit]

    report, per_example = evaluate(model, test_examples, dcfg)
    print(json.dumps(report, indent=2))

    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w") as f:
        json.dump({"summary": report, "examples": per_example}, f, indent=2)
    print(f"Full report written to {args.report_path}")


if __name__ == "__main__":
    main()
