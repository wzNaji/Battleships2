import streamlit as st
from core.game_logic import player_ships_placement
from core.config import GRID_SIZE, SHIP_LENGTHS

# 0) Initialize session-state buckets    
if "ships" not in st.session_state:
    st.session_state.ships = []              # already placed coords
if "remaining" not in st.session_state:
    st.session_state.remaining = list(SHIP_LENGTHS)
if "reset_cells" not in st.session_state:
    st.session_state.reset_cells = False

# 1) Clear cells if flagged
if st.session_state.reset_cells:
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            st.session_state[f"cell_{r}_{c}"] = False
    st.session_state.reset_cells = False
    st.rerun()

st.title("Battleship Placement – 7×7")

if not st.session_state.remaining:
    st.success("✅ All ships placed! Ready to battle.")
    st.stop()

next_len = st.session_state.remaining[0]
st.write(f"Select exactly **{next_len}** cells for your next ship.")

# 2) Draw grid & collect this‐turn selections
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

# 3) Place Ship button, only enabled when count matches next_len
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
