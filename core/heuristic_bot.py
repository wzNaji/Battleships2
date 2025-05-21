import random
import numpy as np
from core.config import SHIP_LENGTHS, GRID_SIZE

Coordinate = tuple[int,int]
Ships_dt = list[list[Coordinate]]

ships : Ships_dt = []

# def place_ships_for_bot(grid_size: int, ship_lenghts: list[int]) -> tuple[np.ndarray, list[list[Coordinate]]]: