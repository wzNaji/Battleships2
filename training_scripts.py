'''
train_tf.py: Standalone DQN training script for Battleship using TensorFlow.
All game logic and state are self-contained—no external GameState module needed.
Usage: python train_tf.py
'''

import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, optimizers

# --- Configuration ---
GRID_SIZE = 7
SHIP_LENGTHS = [4, 3, 2]

# --- Game state management ---
class GameState:
    """Manages Battleship game state without external dependencies."""
    def __init__(self):
        self.reset_all()

    def reset_all(self):
        # Player side
        self.ships = []
        self.player_hits_opponent = set()
        self.player_misses_opponent = set()
        # Opponent side
        self.opponent_ships = []
        self.opponent_hits_player = set()
        self.opponent_misses_player = set()
        # Place opponent ships
        self.place_opponent_ships()

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
                # check overlap
                if not any(coord in cell for cell in self.opponent_ships for coord in coords):
                    self.opponent_ships.append(coords)
                    placed = True

# --- Game logic helper functions ---
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
        r, c = divmod(int(action), GRID_SIZE)
        coord = (r, c)
        # Player move
        reward=0 #initialise
        if any(coord in ship for ship in self.state.opponent_ships):
            self.state.player_hits_opponent.add(coord)
            reward += 2.0
            if is_single_opponent_ship_sunken(self.state, coord):
                reward += 10 ###len(SHIP_LENGTHS)
        else:
            self.state.player_misses_opponent.add(coord)
            reward = -0.1
        if all_opponent_ships_sunk(self.state):
            reward += 20.0
            self.done = True
        # Opponent random
        if not self.done:
            choices = [(i,j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)
                       if (i,j) not in self.state.opponent_hits_player
                       and (i,j) not in self.state.opponent_misses_player]
            opp = random.choice(choices)
            if any(opp in ship for ship in self.state.ships):
                self.state.opponent_hits_player.add(opp)
            else:
                self.state.opponent_misses_player.add(opp)
            if all_player_ships_sunk(self.state):
                reward -= 10.0
                self.done = True
        return self._get_obs(), reward, self.done, {}

# --- Hyperparameters & Setup ---
grid_cells = GRID_SIZE * GRID_SIZE
epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.1, 50000
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
    state = env.reset()
    total_reward = 0.0
    for t in range(max_steps):
        eps = get_epsilon(steps_done)
        if random.random() < eps:
            action = random.randrange(grid_cells)
        else:
            obs = tf.expand_dims(state, 0)
            action = int(tf.argmax(policy_net(obs)[0]).numpy())
        next_state, reward, done, _ = env.step(action)
        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps_done += 1
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
        print(f"Episode {ep}, Reward: {total_reward:.2f}")

policy_net.save('battleship_dqn_tf.keras')
print("Training complete, model saved to 'battleship_dqn_tf'.")
