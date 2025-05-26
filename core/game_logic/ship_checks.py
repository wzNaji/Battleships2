# core/ship_checks.py

import streamlit as st
from core.config import GRID_SIZE, SHIP_LENGTHS
from typing import Tuple, List

Coordinate = Tuple[int,int]
Ships = List[List[Coordinate]]

def is_single_opponent_ship_sunken(coord: Coordinate) -> bool:
    comp_ships = st.session_state.opponent_ships
    hits = st.session_state.player_hits_opponent
    
    # Iterate through ships using an index
    for i, ship in enumerate(comp_ships):
        if coord in ship:
            cells = all(cell in hits for cell in ship)  # Check if all cells are hit

            if cells:
                comp_ships.pop(i)  # Remove the ship
                st.session_state.opponent_ships = comp_ships #Update the board
                print(comp_ships)
                return True #Ship is sunk
            else:
                return False #Ship isn't sunk
    raise ValueError(f"No ship occupies {coord}")


def all_opponent_ships_sunk() -> bool:
    comp_ships = st.session_state.opponent_ships
    hits       = st.session_state.player_hits_opponent
    return all(all(cell in hits for cell in ship) for ship in comp_ships)

def is_single_player_ship_sunken(coord: Coordinate) -> bool:
    player_ships = st.session_state.ships
    hits         = st.session_state.opponent_hits_player
    for ship in player_ships:
        if coord in ship:
            return all(cell in hits for cell in ship)
    raise ValueError(f"No ship occupies {coord}")

def all_player_ships_sunk() -> bool:
    player_ships = st.session_state.ships
    hits         = st.session_state.opponent_hits_player
    return all(all(cell in hits for cell in ship) for ship in player_ships)
