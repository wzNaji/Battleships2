import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models

def create_dqn_model(input_shape, num_actions):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_actions, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

def get_state(hits, misses, grid_size):
    """
    Returns a flattened numpy array representing the current game state.
    Hits are encoded as 1, misses as -1, and unknown cells as 0.
    """
    state = np.zeros((grid_size, grid_size))
    for (x, y) in hits:
        state[x, y] = 1
    for (x, y) in misses:
        state[x, y] = -1
    return state.flatten()

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.model = create_dqn_model(state_size, action_size)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 2000:
            self.memory.pop(0)

    def replay(self, batch_size=32):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state[np.newaxis], verbose=0)[0])
            target_f = self.model.predict(state[np.newaxis], verbose=0)
            target_f[0][action] = target
            self.model.fit(state[np.newaxis], target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
