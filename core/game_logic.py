import random
import numpy as np
from core.config import SHIP_LENGTHS, GRID_SIZE

Coordinate = tuple[int,int]
Ships_dt = list[list[Coordinate]]

ships : Ships_dt = []
opponent_ships : Ships_dt = []

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
    if length not in SHIP_LENGTHS:
        raise ValueError(f"Invalid ship length {length}; must be one of {SHIP_LENGTHS}.")

    # Validering af hvert coord
    for cell in cells:
        if not (isinstance(cell, tuple) and len(cell) == 2 and
                all(isinstance(v, int) for v in cell)):
            raise ValueError(f"Invalid coordinate: {cell}")

    # Kontrol af ship's coord er på linje og sammenhængende,
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]

    # Fjerner duplicates med set så vi ved om cells er på samme row eller ej
    if len(set(rows)) == 1:
        if sorted(cols) != list(range(min(cols), min(cols) + length)):
            raise ValueError("Horizontal placement must be contiguous")
    elif len(set(cols)) == 1:
        # Vertical: contiguous rows
        if sorted(rows) != list(range(min(rows), min(rows) + length)):
            raise ValueError("Vertical placement must be contiguous.")
    else:
        raise ValueError("Ship must be placed in a straight horizontal or vertical line.")

    # Tjek om skibene overlapper eksisterende skibe
    # for each + generator expression. yielder generator obj ind til "any"
    for existing in ships:
        if any(coord in existing for coord in cells):
            raise ValueError(f"Overlap detected in {cells}.")

    # alt spiller - placer skibet
    ships.append(cells)
    return True

def computer_ships_placement() -> Ships_dt:
    """Randomly place all ships for the computer."""
    

    for length in SHIP_LENGTHS:
        placed = False
        while not placed:
            # Randomly choose horizontal or vertical
            horizontal = random.choice([True, False])

            if horizontal:
                row = random.randint(0, GRID_SIZE - 1)
                col = random.randint(0, GRID_SIZE - length)
                coords = [(row, col + i) for i in range(length)]
            else:
                row = random.randint(0, GRID_SIZE - length)
                col = random.randint(0, GRID_SIZE - 1)
                coords = [(row + i, col) for i in range(length)]

            # Check overlap with any existing ship
            overlap = any(
                coord in existing
                for existing in opponent_ships
                for coord in coords
            )
            if not overlap:
                opponent_ships.append(coords)
                placed = True

    return opponent_ships