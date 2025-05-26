import streamlit as st
from core.config import GRID_SIZE
from core.game_logic.ship_checks import all_opponent_ships_sunk, is_single_opponent_ship_sunken


def render_player_grid():
    if st.session_state.end_game_message == "U WON!":
        return

    ships = st.session_state.get("ships")
    hits = st.session_state.get("opponent_hits_player")
    misses = st.session_state.get("opponent_misses_player")

    for r in range(GRID_SIZE):
        cols = st.columns(GRID_SIZE)
        for c, col in enumerate(cols):
            coord = (r, c)

            # Default symbol
            symbol = "?"

            # Check if cell contains a ship
            if any(coord in ship for ship in ships):
                symbol = "🚢"

            # Override if the opponent hit this cell
            if coord in hits:
                symbol = "🔥"
                # Måske target func her?
            # Override if the opponent missed
            elif coord in misses:
                symbol = "○"

            col.markdown(symbol)



def render_opponent_grid():
    if st.session_state.end_game_message == "U WON!":
        return

    hits = st.session_state.get("player_hits_opponent")
    misses = st.session_state.get("player_misses_opponent")
    ships = st.session_state.get("opponent_ships")

    for r in range(GRID_SIZE):
        cols = st.columns(GRID_SIZE)
        for c, col in enumerate(cols):
            coord = (r, c)
            if coord in hits:
                col.markdown("🔥")
            elif coord in misses:
                col.markdown("○")
            else:
                if col.button("?", key=f"fire_{r}_{c}"):
                    if any(coord in ship for ship in ships):
                        st.session_state.player_hits_opponent.add(coord)
                        if is_single_opponent_ship_sunken(coord):
                            print("skib sunket")
                    else:
                        st.session_state.player_misses_opponent.add(coord)


                    if all_opponent_ships_sunk():
                        st.session_state.end_game_message = "U WON!"
                        if not st.session_state.game_over_rerun_done:
                            st.session_state.game_over_rerun_done = True                           
                            st.rerun()
                    # hand off to computer
                    st.session_state.current_turn = "computer"
                    st.rerun()
                    return
