"""Render one or more grid instances with the BFS-optimal path and (optionally)
a model's path overlaid, saved as a PNG for a quick visual sanity check.

Usage:
    python -m src.visualize --config configs/default.yaml --index 0
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from src.env import GridInstance, apply_moves
from src.prompts import parse_moves


def plot_instance(inst: GridInstance, optimal_moves, model_moves=None, title="", save_path="out.png"):
    fig, ax = plt.subplots(figsize=(5, 5))
    grid = [[0] * inst.size for _ in range(inst.size)]
    for (r, c) in inst.obstacle_set:
        grid[r][c] = 1
    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1, origin="upper")

    def path_to_xy(moves):
        result = apply_moves(inst, moves)
        rs = [p[0] for p in result.visited]
        cs = [p[1] for p in result.visited]
        return cs, rs  # x=col, y=row

    xs, ys = path_to_xy(optimal_moves)
    ax.plot(xs, ys, "-o", color="tab:green", label=f"BFS optimal ({len(optimal_moves)} steps)",
            linewidth=2, markersize=4)

    if model_moves is not None:
        xs2, ys2 = path_to_xy(model_moves)
        ax.plot(xs2, ys2, "--x", color="tab:red", label=f"Model ({len(model_moves)} steps)",
                linewidth=2, markersize=6)

    sr, sc = inst.start
    gr, gc = inst.goal
    ax.scatter([sc], [sr], marker="s", color="tab:blue", s=150, label="Start", zorder=5)
    ax.scatter([gc], [gr], marker="*", color="tab:orange", s=250, label="Goal", zorder=5)

    ax.set_xticks(range(inst.size))
    ax.set_yticks(range(inst.size))
    ax.grid(True, color="lightgray", linewidth=0.5)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--eval_report", default=None,
                         help="Optional path to an eval_report.json to overlay the model's actual path.")
    parser.add_argument("--out", default="outputs/example.png")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_path = os.path.join(cfg["data"]["out_dir"], f"{args.split}.jsonl")
    with open(data_path) as f:
        lines = f.readlines()
    ex = json.loads(lines[args.index])
    inst = GridInstance.from_dict(ex["meta"]["instance"])
    optimal_moves = ex["meta"]["optimal_moves"]

    model_moves = None
    if args.eval_report and os.path.exists(args.eval_report):
        with open(args.eval_report) as f:
            report = json.load(f)
        model_moves = report["examples"][args.index]["model_moves"]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plot_instance(inst, optimal_moves, model_moves,
                  title=f"{args.split}[{args.index}] size={inst.size}", save_path=args.out)


if __name__ == "__main__":
    main()
