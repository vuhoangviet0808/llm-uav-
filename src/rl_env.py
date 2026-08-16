"""Gymnasium environment for the UAV point-to-point pathfinding task.

Lets a classical RL algorithm (PPO, via stable-baselines3) be trained and
evaluated on the *exact same* task/test set as the LLM approach, as a
baseline comparison -- similar in spirit to how the original "Learning to
Recharge" paper compares its PPO agent against a Greedy Heuristic, just
adapted to this simplified point-to-point problem (no battery/coverage).

Design choice -- local observation, not a full-grid CNN:
  The original paper's PPO uses a CNN over global+local map layers. To keep
  this baseline simple and robust to implement/train quickly (no custom CNN
  feature extractor plumbing, no image-space edge cases in stable-baselines3
  for tiny 6-12px "images"), the agent instead observes a fixed-size local
  window around itself (obstacles only, egocentric) plus a few scalar
  features giving the relative direction/distance to the goal. This is a
  reactive, goal-directed policy (comparable to a learned "greedy
  heuristic") rather than a full global planner -- a fair but *not*
  identical-power baseline to the paper's CNN architecture. Documented here
  so the comparison isn't overclaimed.

Action space: Discrete(4), mapped to src.env.MOVE_DELTA's order (U/D/L/R).

Reward:
  +1.0   reach goal (episode ends)
  -1.0   invalid move -- hit an obstacle or left the grid (episode ends,
         matching src.env.apply_moves()'s "stop at first invalid move"
         semantics used to score the LLM, so invalid_move_rate is
         comparable across both approaches)
  -0.01  per valid step (encourages shorter paths)
  + potential-based shaping: `shaping_coef * (old_potential - new_potential)`
    -- speeds up learning a goal-directed policy from a partial (local-only)
    observation; provably doesn't change the optimal policy (Ng et al. 1999).

`shaping_mode` picks the potential function:
  "manhattan" (default/original): straight-line distance to the goal. Cheap,
    but *not obstacle-aware* -- right next to a wall/corner it can reward a
    move that walks the agent straight into a dead end, because Manhattan
    distance has no idea an obstacle is in the way.
  "bfs": true shortest-path distance to the goal through the actual
    obstacle layout (one reverse BFS from the goal per episode, using the
    same `MOVE_DELTA`/`is_free` the ground-truth BFS solver uses). This is
    the "LLM-proposed" improvement discussed in chat: a shaping signal that
    always points along a real path, never through a wall -- still a valid
    potential function (doesn't change the optimal policy), just a smarter
    one than Manhattan distance. See src/train_rl.py's --shaping_mode flag
    and notebooks/llm_reward_design.ipynb for the side-by-side comparison.

Episode truncates after `steps_per_size * instance.size` steps if neither
terminal condition is hit (avoids infinite wandering on hard instances).
"""
import random
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.env import MOVE_DELTA, random_instance

ACTIONS = list(MOVE_DELTA.keys())  # ["U", "D", "L", "R"]


class PathfindingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, grid_size_range=(6, 12), obstacle_density_range=(0.10, 0.25),
                 min_optimal_len=3, instances=None, seed=None,
                 local_radius=2, max_size_for_norm=12, steps_per_size=6,
                 shaping_coef=0.1, shaping_mode="manhattan"):
        super().__init__()
        assert shaping_mode in ("manhattan", "bfs"), shaping_mode
        self.shaping_mode = shaping_mode
        self.grid_size_range = tuple(grid_size_range)
        self.obstacle_density_range = tuple(obstacle_density_range)
        self.min_optimal_len = min_optimal_len
        # If a fixed list of GridInstance is given, cycle through it
        # (reproducible replay of the same train set the LLM saw); otherwise
        # sample fresh random instances from the same distribution.
        self.instances = instances
        self._instance_idx = 0
        self._rng = random.Random(seed)

        self.local_radius = local_radius
        self.max_size_for_norm = max_size_for_norm
        self.steps_per_size = steps_per_size
        self.shaping_coef = shaping_coef

        win = 2 * local_radius + 1
        obs_dim = win * win + 4  # local obstacle window + (dx, dy, dist, norm_size)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.inst = None
        self.pos = None
        self.steps = 0
        self.max_episode_steps = None

    def _next_instance(self):
        if self.instances is not None:
            inst = self.instances[self._instance_idx % len(self.instances)]
            self._instance_idx += 1
            return inst
        inst, _ = random_instance(
            size_range=self.grid_size_range,
            obstacle_density=self.obstacle_density_range,
            min_optimal_len=self.min_optimal_len,
            rng=self._rng,
        )
        return inst

    def _manhattan(self, pos):
        gr, gc = self.inst.goal
        return abs(gr - pos[0]) + abs(gc - pos[1])

    def _bfs_distance_map(self):
        """Reverse BFS from the goal over free cells, respecting obstacles --
        gives the TRUE shortest-path distance from every reachable cell to
        the goal, unlike Manhattan distance which ignores the obstacle
        layout entirely."""
        goal = self.inst.goal
        dist = {goal: 0}
        q = deque([goal])
        while q:
            cur = q.popleft()
            for dr, dc in MOVE_DELTA.values():
                nxt = (cur[0] + dr, cur[1] + dc)
                if self.inst.is_free(nxt) and nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        return dist

    def _potential(self, pos):
        if self.shaping_mode == "bfs":
            # every position the agent can legally occupy is reachable from
            # the goal by construction (random_instance() only keeps
            # solvable instances), so this dict lookup should always hit;
            # Manhattan distance is only a defensive fallback.
            return self._dist_map.get(pos, self._manhattan(pos))
        return self._manhattan(pos)

    def _encode(self):
        r0 = self.local_radius
        win = 2 * r0 + 1
        local = np.ones((win, win), dtype=np.float32)  # default: obstacle/out-of-bounds
        pr, pc = self.pos
        for dr in range(-r0, r0 + 1):
            for dc in range(-r0, r0 + 1):
                rr, cc = pr + dr, pc + dc
                if self.inst.in_bounds((rr, cc)):
                    local[dr + r0, dc + r0] = 1.0 if (rr, cc) in self.inst.obstacle_set else 0.0

        gr, gc = self.inst.goal
        dx = (gr - pr) / self.max_size_for_norm
        dy = (gc - pc) / self.max_size_for_norm
        dist = self._manhattan(self.pos) / (2 * self.max_size_for_norm)
        norm_size = self.inst.size / self.max_size_for_norm
        feat = np.array([dx, dy, dist, norm_size], dtype=np.float32)
        return np.concatenate([local.flatten(), feat]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.inst = self._next_instance()
        self.pos = self.inst.start
        self.steps = 0
        self.max_episode_steps = self.steps_per_size * self.inst.size
        if self.shaping_mode == "bfs":
            self._dist_map = self._bfs_distance_map()
        return self._encode(), {}

    def step(self, action):
        move = ACTIONS[int(action)]
        dr, dc = MOVE_DELTA[move]
        npos = (self.pos[0] + dr, self.pos[1] + dc)
        self.steps += 1

        if not self.inst.is_free(npos):
            return self._encode(), -1.0, True, False, {"invalid": True, "move": move}

        old_dist = self._potential(self.pos)
        self.pos = npos
        new_dist = self._potential(self.pos)
        shaping = self.shaping_coef * (old_dist - new_dist)

        if self.pos == self.inst.goal:
            return self._encode(), 1.0, True, False, {"success": True, "move": move}

        truncated = self.steps >= self.max_episode_steps
        return self._encode(), -0.01 + shaping, False, truncated, {"move": move}
