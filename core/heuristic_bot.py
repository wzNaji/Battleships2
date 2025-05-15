import random
import numpy as np
from core.config import SHIP_LENGHTS
from core.config import grid_size, ship_lenghts

Coordinate = tuple[int,int]
Ships = list[list[Coordinate]]

ships : Ships = []

def build_probability_grid(
        computer_guesses: np.ndarray,
        hits: list[tuple[int, int]],
        remaining_lenghts: list[int]
)

def player_ships_placement(cells: list[Coordinate]) -> bool:
    """
    Add a new ship placement to the global 'ships' list.

    Args:
        cells: A list of (row, col) tuples forming a contiguous straight-line ship.
    Returns:
        True if placement succeeded; raises ValueError on invalid placement.
    """
    # kontrol af ship-lenght
    length = len(cells)
    if length not in SHIP_LENGHTS:
        raise ValueError(f"Invalid ship length {length}; must be one of {SHIP_LENGHTS}.")

    # Validering af hvert coord
    for cell in cells:
        if not (isinstance(cell, tuple) and len(cell) == 2 and
                all(isinstance(v, int) for v in cell)):
            raise ValueError(f"Invalid coordinate: {cell}")
    
    # Kontrol af ship's coord er på linje og sammenhængende
    rows = [r for r, _ in cells]
    cols = [c for c, _ in cells]

    # Fjerner duplicates med set så vi ved om cells er på samme row eller ej
    if len(set(rows)) == 1:
        