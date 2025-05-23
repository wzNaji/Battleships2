# Place this after your import statements, e.g.,
# from core.rl_agent import DQNAgent
# from core.game_logic import reset_game, opponent_move
# etc.
import streamlit as st
from utils import get_state
from core.rl_agent import DQNAgent
from game_logic.ship_checks import is_single_opponent_ship_sunken, all_player_ships_sunk
from core.game_logic import reset_game, opponent_move
from config import GRID_SIZE
# Initialize agent once
if 'agent' not in st.session_state:
    st.session_state.agent = DQNAgent(
        state_size=GRID_SIZE * GRID_SIZE,
        action_size=GRID_SIZE * GRID_SIZE
    )

# Example: Run many episodes for training
NUM_EPISODES = 1000  # or more!

for episode in range(1, NUM_EPISODES + 1):
    # Reset game at the start of each episode
    reset_game()
    st.session_state.opponent_hits_player = set()
    st.session_state.opponent_misses_player = set()
    # Initialize any target mode states if needed
    st.session_state.target_mode = False
    st.session_state.target_queue = []
    st.session_state.target_ship_hits = set()
    st.session_state.target_ship_cells = set()

    while True:
        # 1. RL agent's turn
        state = get_state(
            st.session_state.opponent_hits_player,
            st.session_state.opponent_misses_player,
            GRID_SIZE
        )
        action = st.session_state.agent.select_action(state)
        r, c = divmod(action, GRID_SIZE)
        coord = (r, c)
        # execute move, calculate reward
        if any(coord in ship for ship in st.session_state.ships):
            reward = 10
            if is_single_opponent_ship_sunken(coord):
                reward = 20
        else:
            reward = -1

        # Update hits/misses
        if reward > 0:
            st.session_state.opponent_hits_player.add(coord)
        else:
            st.session_state.opponent_misses_player.add(coord)

        # 2. Observe next state
        next_state = get_state(
            st.session_state.opponent_hits_player,
            st.session_state.opponent_misses_player,
            GRID_SIZE
        )

        # 3. Check for game over
        done = all_player_ships_sunk()

        # 4. Store experience
        st.session_state.agent.remember(state, action, reward, next_state, done)

        # 5. Train periodically
        if episode % 10 == 0:
            st.session_state.agent.replay()

        # 6. Save model periodically
        if episode % 50 == 0:
            st.session_state.agent.model.save(f'ml/battleship_train_{episode}.h5')

        if done:
            # Assign big reward for winning
            print(f"Episode {episode} complete: RL wins.")
            break

        # 7. Opponent's move (your existing function)
        opponent_move()
        if all_player_ships_sunk():
            # Opponent won this episode; optional: record losses
            break
