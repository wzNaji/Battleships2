import random
import tensorflow as tf
import numpy as np
import streamlit as st
from core.config import SHIP_LENGTHS, GRID_SIZE
from training_scripts import DQN
from core.game_logic.ship_checks import (
    all_player_ships_sunk,
    is_single_player_ship_sunken,
    is_single_opponent_ship_sunken
)

@st.cache_resource
def load_rl_model():
    model = DQN(GRID_SIZE)
    # build the model
    _ = model(tf.zeros((1, GRID_SIZE, GRID_SIZE), dtype=tf.float32))
    # load your HDF5 weights
    model.load_weights("battleship_dqn.weights.h5")
    return model

Coordinate = tuple[int, int]
Ships_dt = list[list[Coordinate]]

# these hold your placed ships
ships: Ships_dt = []
opponent_ships: Ships_dt = []

def player_ships_placement(cells: list[Coordinate]) -> bool:
    # ... your placement‐validation logic unchanged ...
    length = len(cells)
    if length not in SHIP_LENGTHS:
        raise ValueError(f"Invalid ship length {length}; must be one of {SHIP_LENGTHS}.")
    for cell in cells:
        if not (isinstance(cell, tuple) and len(cell) == 2 and all(isinstance(v, int) for v in cell)):
            raise ValueError(f"Invalid coordinate: {cell}")
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if len(set(rows)) == 1:
        if sorted(cols) != list(range(min(cols), min(cols) + length)):
            raise ValueError("Horizontal placement must be contiguous")
    elif len(set(cols)) == 1:
        if sorted(rows) != list(range(min(rows), min(rows) + length)):
            raise ValueError("Vertical placement must be contiguous.")
    else:
        raise ValueError("Ship must be placed in a straight line.")
    for existing in ships:
        if any(coord in existing for coord in cells):
            raise ValueError(f"Overlap detected in {cells}.")
    ships.append(cells)
    return True


def opponent_ships_placement() -> Ships_dt:
    """Randomly place all ships for the opponent."""
    for length in SHIP_LENGTHS:
        placed = False
        while not placed:
            horizontal = random.choice([True, False])
            if horizontal:
                row = random.randint(0, GRID_SIZE - 1)
                col = random.randint(0, GRID_SIZE - length)
                coords = [(row, col + i) for i in range(length)]
            else:
                row = random.randint(0, GRID_SIZE - length)
                col = random.randint(0, GRID_SIZE - 1)
                coords = [(row + i, col) for i in range(length)]
            overlap = any(
                coord in existing
                for existing in opponent_ships
                for coord in coords
            )
            if not overlap:
                opponent_ships.append(coords)
                placed = True
    return opponent_ships


def create_guesses_grid(hits: set[Coordinate], misses: set[Coordinate], grid_size: int):
    guesses = np.zeros((grid_size, grid_size), dtype=int)
    for x, y in hits:
        guesses[x, y] = 2
    for x, y in misses:
        guesses[x, y] = 1
    return guesses


def simple_probability_grid(guesses, remaining_lengths, grid_size: int):
    prob = np.zeros((grid_size, grid_size), dtype=int)
    for length in remaining_lengths:
        # horizontal
        for r in range(grid_size):
            for c in range(grid_size - length + 1):
                span = guesses[r, c : c + length]
                if 1 in span:
                    continue
                for i in range(length):
                    if guesses[r, c + i] == 0:
                        prob[r, c + i] += 1
        # vertical
        for c in range(grid_size):
            for r in range(grid_size - length + 1):
                span = guesses[r : r + length, c]
                if 1 in span:
                    continue
                for i in range(length):
                    if guesses[r + i, c] == 0:
                        prob[r + i, c] += 1
    return prob


def get_next_guess(grid_size: int, remaining_lengths: list[int]) -> Coordinate:
    hits = st.session_state.opponent_hits_player
    misses = st.session_state.opponent_misses_player
    grid = create_guesses_grid(hits, misses, grid_size)
    prob_grid = simple_probability_grid(grid, remaining_lengths, grid_size)
    candidates = list(zip(*np.where(prob_grid == prob_grid.max())))
    return random.choice(candidates)


def enqueue_neighbors(coord: Coordinate):
    """Add orthogonal neighbors of `coord` to target_queue if valid and untried."""
    r, c = coord
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        neighbor = (nr, nc)
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            if (neighbor not in st.session_state.opponent_hits_player
               and neighbor not in st.session_state.opponent_misses_player
               and neighbor not in st.session_state.target_queue):
                st.session_state.target_queue.append(neighbor)


def opponent_move():
    """AI turn: hunt or target mode, then fire and update state."""
    # 1) pick coordinate
    if st.session_state.target_mode and st.session_state.target_queue:
        coord = st.session_state.target_queue.pop(0)
    else:
        # exit target mode if queue exhausted
        st.session_state.target_mode = False
        coord = get_next_guess(GRID_SIZE, SHIP_LENGTHS)

    # 2) fire
    if any(coord in ship for ship in st.session_state.ships):
        st.session_state.opponent_hits_player.add(coord)
        # first hit on a new ship? initialize target-mode state
        if not st.session_state.target_mode:
            st.session_state.target_mode = True
            st.session_state.target_ship_hits = {coord}
            # identify full ship cells
            for ship in st.session_state.ships:
                if coord in ship:
                    st.session_state.target_ship_cells = set(ship)
                    break
        else:
            # continuing target
            st.session_state.target_ship_hits.add(coord)
        # enqueue neighbors for further targeting
        st.session_state.target_queue.clear()
        enqueue_neighbors(coord)
        # if that ship is now sunk, clear target-mode data
        if is_single_player_ship_sunken(coord):
            cells = is_single_player_ship_sunken(coord)
            st.session_state.target_mode = False
            st.session_state.target_queue.clear()
            st.session_state.target_ship_hits.clear()
            st.session_state.target_ship_cells.clear()
    else:
        st.session_state.opponent_misses_player.add(coord)

    # 3) check for defeat
    if all_player_ships_sunk():
        st.session_state.end_game_message = "COMPUTER WINS!"


def reset_game():
    # clear out everything in session_state
    st.session_state.clear()
    ships.clear()
    opponent_ships.clear()
    # re-run so that all your setdefault(...) calls fire again
    st.rerun()

### ML / AI Functions ###
def rl_agent_move():
    model = load_rl_model()   # cached, loader kun en gang til disk

    # 1) Build the guess grid from the RL agent’s own history
    hits   = st.session_state.opponent_hits_player
    misses = st.session_state.opponent_misses_player

    guesses = create_guesses_grid(hits, misses, GRID_SIZE)
    prob    = simple_probability_grid(guesses, SHIP_LENGTHS, GRID_SIZE)

    # 2) Get Q-values from the model
    q_vals = model.predict(prob[np.newaxis, ...])[0]  # shape (49,)

    # 3) Mask out already‐tried cells
    flat_guesses = guesses.flatten()
    q_vals[flat_guesses != 0] = -np.inf

    # 4) Pick the best action
    action = int(np.argmax(q_vals))
    r, c   = divmod(action, GRID_SIZE)
    coord  = (r, c)

    # 5) “Fire” at the human’s ships
    if any(coord in ship for ship in st.session_state.ships):
        st.session_state.opponent_hits_player.add(coord)
        # (handle sunk if you like)
    else:
        st.session_state.opponent_misses_player.add(coord)

    # 6) Check for human defeat
    if all(
        all(cell in st.session_state.opponent_hits_player for cell in ship)
        for ship in st.session_state.ships
    ):
        st.session_state.end_game_message = "COMPUTER WINS!"

    # 7) Hand control back to the human
    st.session_state.current_turn = "player"
