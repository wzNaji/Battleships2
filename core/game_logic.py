import random
import numpy as np
import streamlit as st
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

def opponent_ships_placement() -> Ships_dt:
    """Randomly place all ships for the opponent."""
    

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

def create_guesses_grid(hits, misses, grid_size):
    guesses = np.zeros((grid_size, grid_size), dtype=int)
    for (x, y) in hits:
        guesses[x, y] = 2  # Hit
    for (x, y) in misses:
        guesses[x, y] = 1  # Miss
    return guesses

def simple_probability_grid(guesses, remaining_lengths, grid_size):
    prob = np.zeros((grid_size, grid_size), dtype=int)
    for length in remaining_lengths:
        # Horizontal
        for r in range(grid_size):
            for c in range(grid_size - length + 1):
                span = guesses[r, c:c+length]
                if 1 in span:
                    continue
                for i in range(length):
                    if guesses[r, c+i] == 0:
                        prob[r, c+i] += 1
        # Vertical
        for c in range(grid_size):
            for r in range(grid_size - length + 1):
                span = guesses[r:r+length, c]
                if 1 in span:
                    continue
                for i in range(length):
                    if guesses[r+i, c] == 0:
                        prob[r+i, c] += 1
    return prob

hits = st.session_state.get("player_hits_opponent")
misses = st.session_state.get("player_misses_opponent")

guesses = create_guesses_grid(hits, misses, GRID_SIZE)
probability_grid = simple_probability_grid(guesses, SHIP_LENGTHS, GRID_SIZE)

# Print the probability values in a readable 2D array format
print("Probability grid (value for each cell):")
for row in probability_grid:
    print(' '.join(f"{val:2d}" for val in row))
st.write(st.session_state.end_game_message)

def is_single_opponent_ship_sunken(coord):
    """
    Return True if the ship containing `coord` is fully hit,
    False if it’s been hit but not yet sunk,
    and raise if no ship occupies that coord.
    """
    comp_ships = st.session_state.opponent_ships
    hits = st.session_state.player_hits_opponent

    for ship in comp_ships:
        if coord in ship:
            # Once we know this is the right ship, return whether every cell is hit
            return all(cell in hits for cell in ship)

    # Only if no ship ever contained coord do we error
    raise ValueError(f"No ship occupies cell {coord}")


def all_opponent_ships_sunk():
    comp_ships = st.session_state.opponent_ships
    hits = st.session_state.player_hits_opponent

    for ship in comp_ships:
        # Check if each ship is fully hit
        if not all(cell in hits for cell in ship):
            return False  # Found a ship not yet sunk
    return True  # All ships are sunk

