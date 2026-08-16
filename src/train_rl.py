"""Train a PPO agent (stable-baselines3) on the UAV pathfinding task -- a
classical-RL baseline to compare against the LLM approach, in the same
spirit as the original "Learning to Recharge" paper comparing its PPO agent
against a Greedy Heuristic (here: PPO vs. zero-shot/fine-tuned LLM).

Trains on the *same* train.jsonl instances the LLM was fine-tuned on
(cycled through repeatedly -- PPO is on-policy online RL, not one-shot SFT,
so instances are replayed many times over `--timesteps` environment steps).

Usage:
    python -m src.train_rl --config configs/default.yaml --timesteps 300000
"""
import argparse
import json
import os

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env import GridInstance
from src.rl_env import PathfindingEnv


def load_instances(path):
    instances = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            instances.append(GridInstance.from_dict(ex["meta"]["instance"]))
    return instances


def make_env(dcfg, instances, seed):
    def _init():
        env = PathfindingEnv(
            grid_size_range=dcfg["grid_size_range"],
            obstacle_density_range=dcfg["obstacle_density_range"],
            min_optimal_len=dcfg.get("min_optimal_len", 3),
            max_size_for_norm=dcfg["grid_size_range"][1],
            instances=instances,
            seed=seed,
        )
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--output_dir", default="outputs/ppo-pathfinding")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg["data"]

    train_instances = load_instances(os.path.join(dcfg["out_dir"], "train.jsonl"))
    print(f"Loaded {len(train_instances)} training instances (same ones the LLM "
          f"fine-tuned on) -- cycling through them across {args.timesteps} env steps.")

    env = DummyVecEnv([make_env(dcfg, train_instances, seed=i) for i in range(args.n_envs)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[128, 128]),
    )
    model.learn(total_timesteps=args.timesteps)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save(os.path.join(args.output_dir, "ppo_model"))
    print(f"Saved PPO model to {args.output_dir}/ppo_model.zip")


if __name__ == "__main__":
    main()
