# core/utils.py
import numpy as np

def get_state(hits, misses, grid_size):
    """
    Encodes hits and misses into a 1D numpy array.
    Hits = 1, misses = -1, unknown = 0.
    """
    state = np.zeros((grid_size, grid_size))
    for (x, y) in hits:
        state[x, y] = 1
    for (x, y) in misses:
        state[x, y] = -1
    return state.flatten()