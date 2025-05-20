import streamlit as st
from core.game_logic import player_ships_placement, computer_ships_placement
from core.config import GRID_SIZE, SHIP_LENGTHS
from core.render import render_opponent_grid, render_player_grid

# Initialize session-state buckets
if "ships" not in st.session_state:
    st.session_state.ships = []              # already placed coords
if "remaining" not in st.session_state:
    st.session_state.remaining = list(SHIP_LENGTHS)
if "reset_cells" not in st.session_state:
    st.session_state.reset_cells = False
if "computer_ships" not in st.session_state:
    st.session_state.computer_ships = computer_ships_placement()
    st.session_state.player_hits_opponent = set()
    st.session_state.player_misses_opponent = set()
    st.session_state.turn = "player"  # placement phase

# Clear cells if flagged
if st.session_state.reset_cells:
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            st.session_state[f"cell_{r}_{c}"] = False
    st.session_state.reset_cells = False
    st.rerun()

st.title("Battleship – 7×7")

# Switch to battle mode once all ships are placed
if not st.session_state.remaining:
    st.session_state.turn = "battle"

# Battle phase: early exit after rendering
if st.session_state.turn == "battle":
    st.header("Fire at the enemy!")
    render_opponent_grid()
    st.write(" ")
    st.write("Your board")
    render_player_grid()
    st.stop()

# ----------------------------------------- Placement phase -----------------------------------------------------
next_len = st.session_state.remaining[0]
st.write(f"Select exactly **{next_len}** cells for your next ship.")

# Draw grid & collect this-turn selections
selected_cells: list[tuple[int,int]] = []
for r in range(GRID_SIZE):
    cols = st.columns(GRID_SIZE)
    for c, col in enumerate(cols):
        occupied = any((r, c) in ship for ship in st.session_state.ships)
        if occupied:
            col.markdown("🚢")
        else:
            key = f"cell_{r}_{c}"
            if col.checkbox(f"Select cell {r},{c}", key=key, label_visibility="collapsed"):
                selected_cells.append((r, c))

# Place Ship button, only enabled when count matches next_len
can_place = len(selected_cells) == next_len
if st.button("Place Ship", disabled=not can_place):
    try:
        player_ships_placement(selected_cells)
        st.session_state.ships.append(list(selected_cells))
        st.session_state.remaining.pop(0)
        st.success(f"Placed {next_len}-cell ship at {selected_cells}")
        st.session_state.reset_cells = True

        st.rerun()
    except Exception as e:
        st.error(str(e))
