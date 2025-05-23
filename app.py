import streamlit as st
from core.utils import get_state
from core.rl_agent import DQNAgent
from core.config import GRID_SIZE, SHIP_LENGTHS
from core.game_logic.ship_checks import is_single_player_ship_sunken, all_player_ships_sunk
from core.game_logic.game_logic import (
    player_ships_placement,
    opponent_ships_placement,
    opponent_move,
)
from core.render import render_opponent_grid, render_player_grid

# --- Session‐state initialization ---
st.session_state.setdefault("game_over_rerun_done", False)
st.session_state.setdefault("current_turn", "player")
st.session_state.setdefault("ships", [])
st.session_state.setdefault("remaining", list(SHIP_LENGTHS))
st.session_state.setdefault("placement_turn", "placement")
st.session_state.setdefault("end_game_message", "")
st.session_state.setdefault("player_hits_opponent", set())
st.session_state.setdefault("player_misses_opponent", set())
st.session_state.setdefault("opponent_hits_player", set())
st.session_state.setdefault("opponent_misses_player", set())

# --- Target-mode AI state ---
st.session_state.setdefault("target_mode", False)
st.session_state.setdefault("target_queue", [])        # queued coords to try in target mode
st.session_state.setdefault("target_ship_hits", set())  # coords hit on the current target ship
st.session_state.setdefault("target_ship_cells", set()) # all coords of current target ship

# Place opponent ships once
if not st.session_state.get("opponent_ships"):
    st.session_state.opponent_ships = opponent_ships_placement()

if 'agent' not in st.session_state:
    state_size = GRID_SIZE * GRID_SIZE  # e.g., 7*7=49
    action_size = GRID_SIZE * GRID_SIZE
    st.session_state.agent = DQNAgent(state_size, action_size)

if 'learning_counter' not in st.session_state:
    st.session_state.learning_counter = 0

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
def rl_agent_opponent_move():
    agent = st.session_state.agent
    
    # 1. Get current game state as a vector
    state = get_state(st.session_state.opponent_hits_player,
                      st.session_state.opponent_misses_player,
                      GRID_SIZE)
    
    # 2. RL agent selects an action
    action = agent.select_action(state)
    
    # 3. Convert to grid coordinate
    r, c = divmod(action, GRID_SIZE)
    coord = (r, c)
    
    # 4. Execute move
    if any(coord in ship for ship in st.session_state.ships):
        reward = 10
        if is_single_player_ship_sunken(coord):
            reward = 20
        st.session_state.opponent_hits_player.add(coord)
    else:
        reward = -1
        st.session_state.opponent_misses_player.add(coord)
    
    if all_player_ships_sunk():
        # heavily reward the model for winning
        reward = 100  # or higher, to emphasize winning
    # 5. Observe next state
    next_state = get_state(st.session_state.opponent_hits_player,
                           st.session_state.opponent_misses_player,
                           GRID_SIZE)
    
    # 6. Check for game over
    done = all_player_ships_sunk()
    
    # 7. Store experience
    agent.remember(state, action, reward, next_state, done)
    
    # 8. Increment counter
    st.session_state.learning_counter += 1
    
    # 9. Periodically train
    if st.session_state.learning_counter % 10 == 0:
        agent.replay()
    
    # 10. Save model periodically
    if st.session_state.learning_counter % 50 == 0:
        agent.model.save('ml/battleship_dqn.h5')
    
    # 11. Handle game end
    if done:
        st.session_state.end_game_message = "COMPUTER WINS!"
# ----------------------------------- Battle Phase -----------------------------------
# 1) Player turn: render opponent grid (fires), then player grid
render_opponent_grid()
st.write(" ")
if st.session_state.end_game_message != "U WON!":
    st.write("Your board")
    st.header("Fleet status")
render_player_grid()

# 2) Computer turn
if st.session_state.current_turn == "computer" and not st.session_state.end_game_message:
    rl_agent_opponent_move()
    st.session_state.current_turn = "player"
    st.rerun()

st.stop()

