import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, optimizers

# --- Configuration ---
GRID_SIZE = 7
SHIP_LENGTHS = [4, 3, 2]
Coordinate = tuple[int, int]
Ships_dt = list[list[Coordinate]]

# --- Helper functions for AI targeting ---

def create_guesses_grid(hits: set[Coordinate], misses: set[Coordinate], grid_size: int):
    guesses = np.zeros((grid_size, grid_size), dtype=int)
    for x, y in hits:
        guesses[x, y] = 2
    for x, y in misses:
        guesses[x, y] = 1
    return guesses

def simple_probability_grid(guesses, remaining_lengths, grid_size: int):
    prob = np.zeros((grid_size, grid_size), dtype=int)
    for length in remaining_lengths:
        # horizontal
        for r in range(grid_size):
            for c in range(grid_size - length + 1):
                span = guesses[r, c : c + length]
                if 1 in span:
                    continue
                for i in range(length):
                    if guesses[r, c + i] == 0:
                        prob[r, c + i] += 1
        # vertical
        for c in range(grid_size):
            for r in range(grid_size - length + 1):
                span = guesses[r : r + length, c]
                if 1 in span:
                    continue
                for i in range(length):
                    if guesses[r + i, c] == 0:
                        prob[r + i, c] += 1
    return prob

def get_next_guess(s, grid_size: int, remaining_lengths: list[int], hits, misses) -> Coordinate:
    grid = create_guesses_grid(hits, misses, grid_size)
    prob_grid = simple_probability_grid(grid, remaining_lengths, grid_size)
    candidates = list(zip(*np.where(prob_grid == prob_grid.max())))
    return random.choice(candidates)

def enqueue_neighbors(coord, grid_size, hits, misses, queue):
    """
    Enqueue orthogonal neighbors of a hit for target mode.
    """
    r, c = coord
    for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            if (nr, nc) not in hits and (nr, nc) not in misses and (nr, nc) not in queue:
                queue.append((nr, nc))

# --- Game state management ---
class GameState:
    """Manages Battleship game state, including AI targeting state."""
    def __init__(self):
        self.reset_all()

    def reset_all(self):
        # Player side
        self.ships = []
        self.player_hits_opponent = set()
        self.player_misses_opponent = set()
        # Opponent side (AI)
        self.opponent_ships = []
        self.opponent_hits_player = set()
        self.opponent_misses_player = set()
        # Sink bonus tracking for player
        self.sunk_ships_awarded_opponent = set()
        # Heuristic opponent targeting state
        self.opponent_target_mode = False
        self.opponent_target_queue = []
        self.opponent_target_hits = set()
        self.opponent_target_cells = set()
        # RL agent exploration targetting state
        self.rl_agent_target_mode = False
        self.rl_agent_target_queue = []
        self.rl_agent_target_hits = set()
        self.rl_agent_target_cells = set()

        # Place ships
        self.place_opponent_ships()
        self.place_player_ships()

    def place_opponent_ships(self):
        self.opponent_ships.clear()
        for length in SHIP_LENGTHS:
            placed = False
            while not placed:
                horizontal = random.choice([True, False])
                if horizontal:
                    r = random.randint(0, GRID_SIZE - 1)
                    c = random.randint(0, GRID_SIZE - length)
                    coords = [(r, c + i) for i in range(length)]
                else:
                    r = random.randint(0, GRID_SIZE - length)
                    c = random.randint(0, GRID_SIZE - 1)
                    coords = [(r + i, c) for i in range(length)]
                if not any(coord in cell for cell in self.opponent_ships for coord in coords):
                    self.opponent_ships.append(coords)
                    placed = True

    def place_player_ships(self):
        self.ships.clear()
        for length in SHIP_LENGTHS:
            placed = False
            while not placed:
                horizontal = random.choice([True, False])
                if horizontal:
                    r = random.randint(0, GRID_SIZE - 1)
                    c = random.randint(0, GRID_SIZE - length)
                    coords = [(r, c + i) for i in range(length)]
                else:
                    r = random.randint(0, GRID_SIZE - length)
                    c = random.randint(0, GRID_SIZE - 1)
                    coords = [(r + i, c) for i in range(length)]
                if not any(coord in cell for cell in self.ships for coord in coords):
                    self.ships.append(coords)
                    placed = True



# --- ship checks logic helper functions ---

def is_single_opponent_ship_sunken(state: GameState, coord: tuple[int,int]) -> bool:
    for ship in state.opponent_ships:
        if coord in ship:
            return all(cell in state.player_hits_opponent for cell in ship)
    raise ValueError(f"No opponent ship occupies {coord}")


def all_opponent_ships_sunk(state: GameState) -> bool:
    return all(all(cell in state.player_hits_opponent for cell in ship)
               for ship in state.opponent_ships)


def is_single_player_ship_sunken(state: GameState, coord: tuple[int,int]) -> bool:
    for ship in state.ships:
        if coord in ship:
            return all(cell in state.opponent_hits_player for cell in ship)
    raise ValueError(f"No player ship occupies {coord}")


def all_player_ships_sunk(state: GameState) -> bool:
    return all(all(cell in state.opponent_hits_player for cell in ship)
               for ship in state.ships)



# --- DQN model and replay buffer ---
class DQN(tf.keras.Model):
    def __init__(self, grid_size, hidden=128):
        super().__init__()
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(hidden, activation='relu')
        self.d2 = layers.Dense(hidden, activation='relu')
        self.out = layers.Dense(grid_size * grid_size)

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.out(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)


# --- Environment wrapper ---
class BattleshipEnv:
    def __init__(self):
        self.state = GameState()
        self.done = False

    def reset(self):
        self.state.reset_all()
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for r,c in self.state.player_hits_opponent:
            grid[r,c] = 1.0
        for r,c in self.state.player_misses_opponent:
            grid[r,c] = -1.0
        return grid

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode done; call reset().")
        #from 1D array to 2D array
        r, c = divmod(int(action), GRID_SIZE)
        coord = (r, c)
        # Prevent duplicate actions
        if coord in self.state.player_hits_opponent or coord in self.state.player_misses_opponent:
            return self._get_obs(), -100, self.done, {}

        reward = 0.0

        # --- Player move ---
        hit = any(coord in ship for ship in self.state.opponent_ships)
        if hit:
            self.state.player_hits_opponent.add(coord)
            reward += 1.0
            self.rl_agent_target_mode = True
            # Check and award sunk bonus once
            for idx, ship in enumerate(self.state.opponent_ships):
                if coord in ship and idx not in self.state.sunk_ships_awarded_opponent:
                    if all(cell in self.state.player_hits_opponent for cell in ship):
                        reward += 2
                        self.state.sunk_ships_awarded_opponent.add(idx)
                    break
        else:
            self.state.player_misses_opponent.add(coord)
            reward -= 0.2

        # Check victory for player
        if all_opponent_ships_sunk(self.state):
            reward += 10.0
            self.done = True
            return self._get_obs(), reward, self.done, {}

        # --- Opponent move (hunt & target) ---
        s = self.state
        # 1) select shot
        if s.opponent_target_mode and s.opponent_target_queue:
            opp_coord = s.opponent_target_queue.pop(0)
        else:
            s.opponent_target_mode = False
            opp_coord = get_next_guess(
                s,
                GRID_SIZE,
                SHIP_LENGTHS,
                hits=s.opponent_hits_player,
                misses=s.opponent_misses_player,
            )

        # 2) fire
        if any(opp_coord in ship for ship in s.ships):
            s.opponent_hits_player.add(opp_coord)
            # enter or continue target mode
            if not s.opponent_target_mode:
                s.opponent_target_mode = True
                s.opponent_target_hits = {opp_coord}
                # identify ship cells
                for ship in s.ships:
                    if opp_coord in ship:
                        s.opponent_target_cells = set(ship)
                        break
            else:
                s.opponent_target_hits.add(opp_coord)

            # enqueue neighbors
            s.opponent_target_queue.clear()
            enqueue_neighbors(
                opp_coord,
                GRID_SIZE,
                s.opponent_hits_player,
                s.opponent_misses_player,
                s.opponent_target_queue
            )

            # check if sunk
            sunk = is_single_player_ship_sunken(self.state, opp_coord)
            if sunk:
                s.opponent_target_mode = False
                s.opponent_target_queue.clear()
                s.opponent_target_hits.clear()
                s.opponent_target_cells.clear()
        else:
            s.opponent_misses_player.add(opp_coord)

        # 3) check defeat for opponent
        if all_player_ships_sunk(self.state):
            reward -= 10.0
            self.done = True
        return self._get_obs(), reward, self.done, {}

# --- Hyperparameters & Setup ---
grid_cells = GRID_SIZE * GRID_SIZE
epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.2, 200_000
gamma = 0.99
lr = 1e-3
batch_size = 64
memory_size = 100000
num_episodes = 10000
max_steps = GRID_SIZE * GRID_SIZE
update_target_every = 100

policy_net = DQN(GRID_SIZE)
target_net = DQN(GRID_SIZE)
target_net.set_weights(policy_net.get_weights())
optimizer = optimizers.Adam(lr)
memory = ReplayBuffer(memory_size)

# Epsilon schedule
def get_epsilon(step):

    return epsilon_end + (epsilon_start - epsilon_end) * np.exp(-step / epsilon_decay)

# --- Training loop ---
steps_done = 0
for ep in range(num_episodes):
    env = BattleshipEnv()
    self = GameState()
    if all_player_ships_sunk(self) or all_opponent_ships_sunk(self):
        env.reset()
    
    total_reward=0
    
    for t in range(max_steps):
        eps = get_epsilon(steps_done)

        # 1) build your prob & guesses grids
        guesses = create_guesses_grid(
            env.state.player_hits_opponent,
            env.state.player_misses_opponent,
            GRID_SIZE
        )
        prob = simple_probability_grid(guesses, SHIP_LENGTHS, GRID_SIZE)

        # 2) first few episodes: pure heuristic
        if ep < 500:
            candidates = list(zip(*np.where(prob == prob.max())))
            chosen = random.choice(candidates)
            action = int(chosen[0] * GRID_SIZE + chosen[1])

        # 3) afterwards: masked ε-greedy on Q-values
        else:
            # a) get raw Q-values
            q_tensor = policy_net(tf.expand_dims(prob, axis=0))  # shape (1,49)
            q_vals   = q_tensor[0].numpy()                        # shape (49,)

            # b) mask out already-shot cells
            flat_guesses = guesses.flatten()       # 49-long array of {0,1,2}
            valid_idx    = (flat_guesses == 0)     # True == untried
            masked_q     = q_vals.copy()
            masked_q[~valid_idx] = -np.inf

            # c) ε-greedy over the valid indices
            if random.random() < eps:
                choices = np.where(valid_idx)[0]  # list of untried 0–48
                action  = int(random.choice(choices))
            else:
                action  = int(np.argmax(masked_q))

        # 4) fire!
        next_state, reward, done, _ = env.step(action)

        

        memory.push(prob, action, reward, next_state, done)
        prob = next_state
        total_reward += reward
        steps_done += 1

        # Update network
        if len(memory) >= batch_size:
            s_b, a_b, r_b, ns_b, d_b = memory.sample(batch_size)
            with tf.GradientTape() as tape:
                q = policy_net(s_b)
                q_pred = tf.reduce_sum(q * tf.one_hot(a_b, grid_cells), axis=1)
                q_next = target_net(ns_b)
                q_next_max = tf.reduce_max(q_next, axis=1)
                q_target = r_b + (1 - d_b) * gamma * q_next_max
                loss = tf.reduce_mean(tf.square(q_pred - q_target))
            grads = tape.gradient(loss, policy_net.trainable_variables)
            optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))
        
        if done:
            break

    if ep % update_target_every == 0:
        target_net.set_weights(policy_net.get_weights())
        
    print(f"Episode {ep}, Reward: {total_reward:.2f}, Hits: {len(env.state.player_hits_opponent)}, Misses: {len(env.state.player_misses_opponent)}")

policy_net.save('battleship_dqn_tf_fixed.keras')
print("Training complete, model saved to 'battleship_dqn_tf_fixed.keras'.")