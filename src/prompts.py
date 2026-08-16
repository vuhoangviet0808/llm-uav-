"""Prompt / completion templates that turn a GridInstance into text the LLM
sees, and parse the LLM's text output back into a move sequence."""
from typing import List

from src.env import GridInstance, Move, render_grid

SYSTEM_INSTRUCTION = (
    "You control a UAV flying over a grid. Cells are separated by spaces. "
    "'S' is the UAV start, 'G' is the goal, '#' is an obstacle / no-fly zone, "
    "'.' is free space. Output ONLY a sequence of moves separated by spaces, "
    "using U (up), D (down), L (left), R (right), that flies the UAV from S "
    "to G without leaving the grid and without crossing '#'. End your answer "
    "with <END>. Do not output anything else."
)

END_TOKEN = "<END>"


def build_prompt(inst: GridInstance) -> str:
    grid_text = render_grid(inst)
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Grid ({inst.size}x{inst.size}):\n{grid_text}\n\n"
        f"Moves:"
    )


def build_completion(moves: List[Move]) -> str:
    # Leading space so it concatenates cleanly onto a prompt ending in ":"
    return " " + " ".join(moves) + f" {END_TOKEN}"


def parse_moves(text: str) -> List[Move]:
    """Best-effort parse of a model's raw generation into a move list.
    Stops at the first <END> token if present, ignores any other tokens."""
    text = text.split(END_TOKEN)[0]
    tokens = text.strip().split()
    return [t for t in tokens if t in ("U", "D", "L", "R")]
