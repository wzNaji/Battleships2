

ships : list[list[tuple[int,int]]] = []
coordinate = tuple[int,int]

# Cellen har cords fx. (4.2) r = 4, c = 2
def player_ships_placement(cells): 
    if len(cells) <= 2:
        row = cells[0]
        col = cells[1]