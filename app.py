import streamlit as st
from core.config import GRID_SIZE, SHIP_LENGTHS
from core.game_logic.game_logic import (
    player_ships_placement,
    opponent_ships_placement,
    opponent_move,
    reset_game,
    rl_agent_move
)
from core.render import render_opponent_grid, render_player_grid

# --- Session‐state initialization ---

st.session_state.setdefault("game_over_rerun_done", False)
st.session_state.setdefault("current_turn", "player")

st.session_state.setdefault("ships", []) # player ships
st.session_state.setdefault("remaining", list(SHIP_LENGTHS))

st.session_state.setdefault("placement_turn", "placement")
st.session_state.setdefault("end_game_message", "")
st.session_state.setdefault("player_hits_opponent", set())
st.session_state.setdefault("player_misses_opponent", set())
st.session_state.setdefault("opponent_hits_player", set())
st.session_state.setdefault("opponent_misses_player", set())

# --- for choice buttons of the heuristic and the rl_agent ---
st.session_state.setdefault("heuristic_bot", True)
st.session_state.setdefault("rl_agent_bot", False)


# --- Target-mode AI state ---
st.session_state.setdefault("target_mode", False)
st.session_state.setdefault("target_queue", [])        # queued coords to try in target mode
st.session_state.setdefault("target_ship_hits", set())  # coords hit on the current target ship
st.session_state.setdefault("target_ship_cells", set()) # all coords of current target ship

if st.session_state.get("new_game", True):
    # Clear previous game data
    st.session_state['ships'] = []
    st.session_state['player_hits_opponent'] = set()
    st.session_state['player_misses_opponent'] = set()
    st.session_state['opponent_hits_player'] = set()
    st.session_state['opponent_misses_player'] = set()
    st.session_state['end_game_message'] = ""
    
    st.session_state['opponent_ships'] = []
    # Reset placement phase
    st.session_state['placement_turn'] = "placement"
    st.session_state['remaining'] = list(SHIP_LENGTHS)
    st.session_state['current_turn'] = "player"
    # Remove 'new_game' flag after reset
    st.session_state['new_game'] = False
    # valg af knap
    st.session_state['heuristic_bot'] = False
    st.session_state['rl_agent_bot'] = False


# Place opponent ships once
if not st.session_state.get("opponent_ships"):
    st.session_state.opponent_ships = opponent_ships_placement()

st.title("Battleship – 7×7")
if st.session_state.get("end_game_message"):
    st.write(st.session_state.get("end_game_message"))

# -------------------------------- Placement Phase --------------------------------
if st.session_state.placement_turn == "placement":
    next_len = st.session_state.remaining[0]
    st.write(f"Select exactly **{next_len}** cells for your next ship.")

    selected_cells: list[tuple[int,int]] = []
    for r in range(GRID_SIZE):
        cols = st.columns(GRID_SIZE)
        for c, col in enumerate(cols):
            occupied = any((r, c) in ship for ship in st.session_state.ships)
            if occupied:
                col.markdown("🚢")
            else:
                key = f"cell_{r}_{c}"
                if col.checkbox("", key=key, label_visibility="collapsed"):
                    selected_cells.append((r, c))

    can_place = len(selected_cells) == next_len
    if st.button("Place Ship", disabled=not can_place):
        try:
            player_ships_placement(selected_cells)
            st.session_state.ships.append(list(selected_cells))
            st.session_state.remaining.pop(0)
            st.success(f"Placed {next_len}-cell ship at {selected_cells}")
            # clear checkboxes
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    st.session_state.pop(f"cell_{r}_{c}", None)
            if not st.session_state.remaining:
                st.session_state.placement_turn = "battle"
            st.rerun()
        except Exception as e:
            st.error(str(e))
    st.stop()

# ----------------------------------- Battle Phase -----------------------------------
# buttons to choose enemy! basically en switch til state flag
if st.button("Heuristic"):
    st.session_state["heuristic_bot"] = True
    st.session_state["rl_agent_bot"] = False

elif st.button("RL_Agent (decent)"):
    st.session_state["rl_agent_bot"] = True
    st.session_state["heuristic_bot"] = False

# spil mod heuristic hvis knappen til at switch state flag er true
if st.session_state.get("heuristic_bot") and not st.session_state.get("rl_agent_bot"):
    render_opponent_grid()  # indeholder endgame message check før render
    st.write("------------------------------------------------------------------------")
    render_player_grid()    # ^^

    # 2) Computer turn
    if st.session_state.current_turn == "computer" and not st.session_state.end_game_message:
        opponent_move()
        st.session_state.current_turn = "player"
        st.rerun()

    # ——— Reset button ———
    if st.button("🔄 Restart Game"):
        st.session_state['new_game'] = True
        reset_game()

    st.stop()

# spil mod rl_agent hvis knappen til at switch state flag er true
if st.session_state.get("rl_agent_bot") and not st.session_state.get("heuristic_bot"):
    render_opponent_grid()  # indeholder endgame message check før render
    st.write("------------------------------------------------------------------------")
    render_player_grid()    # ^^

    # 2) RL-agent turn
    if st.session_state.current_turn == "computer" and not st.session_state.end_game_message:
        rl_agent_move()        # ← use your trained DQN here
        st.session_state.current_turn = "player"
        st.rerun()

    # ——— Reset button ———
    if st.button("🔄 Restart Game"):
        reset_game()
        st.session_state['new_game'] = True

    st.stop()
