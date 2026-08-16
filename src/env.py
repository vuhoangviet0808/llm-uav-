"""Minimal grid-world UAV pathfinding environment.

This is the simplified problem: a UAV must fly from a start cell S to a goal
cell G on an NxN grid, avoiding obstacles/no-fly-zones '#', without leaving
the grid. No battery, no coverage target, no recharge -- the simplest
version of the UAV planning problem in the "Learning to Recharge" paper.

BFS gives the ground-truth optimal move sequence, used both to build the
fine-tuning dataset and as the reference for evaluation (RPD-style metrics).
"""
import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Move = str  # one of "U", "D", "L", "R"

MOVE_DELTA = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


@dataclass
class GridInstance:
    size: int
    obstacles: List[Tuple[int, int]]
    start: Tuple[int, int]
    goal: Tuple[int, int]

    def __post_init__(self):
        self._obstacle_set = set(tuple(o) for o in self.obstacles)
        self.start = tuple(self.start)
        self.goal = tuple(self.goal)

    @property
    def obstacle_set(self):
        return self._obstacle_set

    def in_bounds(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.size and 0 <= c < self.size

    def is_free(self, pos: Tuple[int, int]) -> bool:
        return self.in_bounds(pos) and pos not in self._obstacle_set

    def to_dict(self):
        return {
            "size": self.size,
            "obstacles": [list(o) for o in self.obstacles],
            "start": list(self.start),
            "goal": list(self.goal),
        }

    @staticmethod
    def from_dict(d):
        return GridInstance(size=d["size"], obstacles=d["obstacles"],
                             start=d["start"], goal=d["goal"])


def bfs_shortest_path(inst: GridInstance) -> Optional[List[Move]]:
    """Shortest move sequence start -> goal, or None if unreachable."""
    start, goal = inst.start, inst.goal
    if start == goal:
        return []
    visited = {start}
    queue = deque([(start, [])])
    while queue:
        pos, path = queue.popleft()
        for move, (dr, dc) in MOVE_DELTA.items():
            npos = (pos[0] + dr, pos[1] + dc)
            if not inst.in_bounds(npos) or npos in inst.obstacle_set or npos in visited:
                continue
            npath = path + [move]
            if npos == goal:
                return npath
            visited.add(npos)
            queue.append((npos, npath))
    return None


def random_instance(size_range=(6, 10), obstacle_density=(0.1, 0.25),
                     min_optimal_len=3, rng: Optional[random.Random] = None,
                     max_tries=500) -> Tuple[GridInstance, List[Move]]:
    """Sample a random solvable instance together with its optimal path."""
    rng = rng or random.Random()
    for _ in range(max_tries):
        size = rng.randint(*size_range)
        density = rng.uniform(*obstacle_density)
        cells = [(r, c) for r in range(size) for c in range(size)]
        n_obstacles = int(size * size * density)
        n_obstacles = min(n_obstacles, len(cells) - 2)
        obstacles = set(rng.sample(cells, n_obstacles)) if n_obstacles > 0 else set()
        free_cells = [c for c in cells if c not in obstacles]
        if len(free_cells) < 2:
            continue
        start, goal = rng.sample(free_cells, 2)
        inst = GridInstance(size=size, obstacles=list(obstacles), start=start, goal=goal)
        path = bfs_shortest_path(inst)
        if path is not None and len(path) >= min_optimal_len:
            return inst, path
    raise RuntimeError("Could not generate a solvable instance within max_tries; "
                        "loosen obstacle_density or min_optimal_len.")


def render_grid(inst: GridInstance) -> str:
    lines = []
    for r in range(inst.size):
        row = []
        for c in range(inst.size):
            pos = (r, c)
            if pos == inst.start:
                row.append("S")
            elif pos == inst.goal:
                row.append("G")
            elif pos in inst.obstacle_set:
                row.append("#")
            else:
                row.append(".")
        lines.append(" ".join(row))
    return "\n".join(lines)


@dataclass
class SimResult:
    final_pos: Tuple[int, int]
    reached_goal: bool
    hit_invalid_at: Optional[int]  # index of first invalid move, if any
    visited: List[Tuple[int, int]] = field(default_factory=list)


def apply_moves(inst: GridInstance, moves: List[Move]) -> SimResult:
    """Simulate a move sequence from inst.start. Stops at the first invalid
    move (out of bounds / obstacle / unknown token)."""
    pos = inst.start
    visited = [pos]
    for i, mv in enumerate(moves):
        if mv not in MOVE_DELTA:
            return SimResult(pos, pos == inst.goal, i, visited)
        dr, dc = MOVE_DELTA[mv]
        npos = (pos[0] + dr, pos[1] + dc)
        if not inst.is_free(npos):
            return SimResult(pos, pos == inst.goal, i, visited)
        pos = npos
        visited.append(pos)
        if pos == inst.goal:
            return SimResult(pos, True, None, visited)
    return SimResult(pos, pos == inst.goal, None, visited)
