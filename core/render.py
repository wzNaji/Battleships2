from core.config import GRID_SIZE
import streamlit as st

def render_player_grid():
    for r in range(GRID_SIZE):
        cols = st.columns(GRID_SIZE)
        for c, col in enumerate(cols):
            col.checkbox(f"{r},{c}", key=f"player_cell_{r}_{c}", label_visibility="collapsed")

def render_opponent_grid():
    for r in range(GRID_SIZE):
        cols = st.columns(GRID_SIZE)
        for c, col in enumerate(cols):
            col.checkbox(f"{r},{c}", key=f"opponent_cell_{r}_{c}", label_visibility="collapsed")
