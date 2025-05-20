import streamlit as st
from core.config import GRID_SIZE

def render_player_grid():
    """
    Shows the player’s ships as 🚢 and empty cells as non-interactive placeholders.
    """
    # Read player ships from session state
    ships = st.session_state.ships
    for r in range(GRID_SIZE):
        cols = st.columns(GRID_SIZE)
        for c, col in enumerate(cols):
            coord = (r, c)
            if any(coord in ship for ship in ships):
                col.markdown("🚢")
            else:
                # placeholder cell
                col.markdown(" ")


def render_opponent_grid():
    """
    Renders the opponent’s grid: untargeted cells are fire buttons; hits show 🔥; misses show ○.
    Clicking a button fires at that coordinate.
    """
    hits = st.session_state.player_hits_opponent
    misses = st.session_state.player_misses_opponent
    comp_ships = st.session_state.computer_ships
    print(comp_ships) # skal fjernes efter kontrol
    for r in range(GRID_SIZE):
        cols = st.columns(GRID_SIZE)
        for c, col in enumerate(cols):
            coord = (r, c)
            if coord in hits:
                col.markdown("🔥")
                if is_single_opponent_ship_sunken(coord): # skal fjernes efter kontrol
                    print("alle ramt pls")
            elif coord in misses:
                col.markdown("○")
            else:
                # Untargeted: offer a fire button
                if col.button("", key=f"fire_{r}_{c}"):
                    if any(coord in ship for ship in comp_ships):
                        st.session_state.player_hits_opponent.add(coord)
                    else:
                        st.session_state.player_misses_opponent.add(coord)
                    
                    st.rerun()
                    

def is_single_opponent_ship_sunken(coord):
    """
    Return True if the ship containing `coord` is fully hit,
    False if it’s been hit but not yet sunk,
    and raise if no ship occupies that coord.
    """
    comp_ships = st.session_state.computer_ships
    hits = st.session_state.player_hits_opponent

    for ship in comp_ships:
        if coord in ship:
            # Once we know this is the right ship, return whether every cell is hit
            return all(cell in hits for cell in ship)

    # Only if no ship ever contained coord do we error
    raise ValueError(f"No ship occupies cell {coord}")


