import numpy as np
import math

ACTIONS = ['forward', 'backward', 'stay']
X_BOUNDS = (-1.2, .6)
VELOCITY_BOUNDS = (-.07, .07)
TERMINAL_X = 0.6
TERMINAL_REWARD = 0
NON_TERMINAL_REWARD = -1
START_X_BOUNDS = (-.6, -0.4)
START_VELOCITY = 0
NUM_TILES = 64
TILING_ROWS_AND_COLS = (8, 8)
TILING_OFFSETS = [
    (0.0, 0.0),
    (1/8, 3/8),
    (2/8, 6/8),
    (3/8, 1/8),
    (4/8, 4/8),
    (5/8, 7/8),
    (6/8, 2/8),
    (7/8, 5/8),
]
NUM_FEATURES = len(TILING_OFFSETS) * NUM_TILES
NUM_ACTION_FEATURES = NUM_FEATURES * len(ACTIONS)

# all inputs are tuples
def get_tile_vector(
    state,
    space_x_range=X_BOUNDS,
    space_y_range=VELOCITY_BOUNDS,
    tile_rows_and_cols=TILING_ROWS_AND_COLS,
    offset=TILING_OFFSETS[0]
    ):
    rows, cols = tile_rows_and_cols
    num_tiles = rows * cols
    tile_width = (space_x_range[1] - space_x_range[0]) / cols
    tile_height = (space_y_range[1] - space_y_range[0]) / rows

    # offset is fraction of a tile in each dimension (e.g., 0.5 = half-tile shift)
    offset_x_world = offset[0] * tile_width
    offset_y_world = offset[1] * tile_height

    x, y = state
    x_min, _ = space_x_range
    y_min, _ = space_y_range

    col = math.floor((x - x_min + offset_x_world) / tile_width)
    row = math.floor((y - y_min + offset_y_world) / tile_height)

    col = max(0, min(cols - 1, col))
    row = max(0, min(rows - 1, row))

    tile_vector = np.zeros(num_tiles)
    tile_vector[row * cols + col] = 1
    return tile_vector
  
def get_feature_vector(state, space_x_range=X_BOUNDS, space_y_range=VELOCITY_BOUNDS, tile_rows_and_cols=TILING_ROWS_AND_COLS, offsets=TILING_OFFSETS):
    feature_vectors = []
    for i, offset in enumerate(offsets):
        feature_vectors.append(get_tile_vector(state, space_x_range, space_y_range, tile_rows_and_cols, offset))
    return np.concatenate(feature_vectors)

def get_action_feature_vector(state, action):
    state_feature_vector = get_feature_vector(state, X_BOUNDS, VELOCITY_BOUNDS, TILING_ROWS_AND_COLS, TILING_OFFSETS)
    action_feature_vector = np.zeros(NUM_ACTION_FEATURES)
    action_index = ACTIONS.index(action)
    start = action_index * NUM_FEATURES
    end = start + NUM_FEATURES
    action_feature_vector[start:end] = state_feature_vector
    return action_feature_vector

def position_update(x_pos, velocity):
    return max(min(x_pos + velocity, X_BOUNDS[1]), X_BOUNDS[0])

def velocity_update(x_pos, velocity, action ):
    if action == 'forward':
        thrust = 1
    elif action == 'backward':
        thrust = -1
    else:
        thrust = 0
    return max(min(velocity + .001 * thrust - .0025 * math.cos(3 * x_pos), VELOCITY_BOUNDS[1]), VELOCITY_BOUNDS[0])

def update_state(state, action):
    new_velocity = velocity_update(state[0], state[1], action)
    new_x_pos = position_update(state[0], new_velocity)
    if new_x_pos == X_BOUNDS[0]:
        new_velocity = 0
    return (new_x_pos, new_velocity)

def get_reward(state):
    if state[0] >= TERMINAL_X:
        return TERMINAL_REWARD
    else:
        return NON_TERMINAL_REWARD
    
def get_next_state_and_reward(state, action):
    new_state = update_state(state, action)
    reward = get_reward(new_state)
    return new_state, reward

def get_start_state():
    return (np.random.uniform(START_X_BOUNDS[0], START_X_BOUNDS[1]), START_VELOCITY)
    
def get_action_value(state, action, w):
    feature_vector = get_action_feature_vector(state, action)
    return np.dot(w, feature_vector)

def get_action_values(state, w):
    state_feature_vector = get_feature_vector(state, X_BOUNDS, VELOCITY_BOUNDS, TILING_ROWS_AND_COLS, TILING_OFFSETS)
    values = []
    for i in range(len(ACTIONS)):
        start = i * NUM_FEATURES
        end = start + NUM_FEATURES
        w_slice = w[start:end]
        values.append(np.dot(w_slice, state_feature_vector))
    return values

def get_next_action(w, state, epsilon):
    action_values = get_action_values(state, w)
    best_action = ACTIONS[np.argmax(action_values)]
    if np.random.random() < epsilon:
        return np.random.choice(ACTIONS)
    else:
        return best_action

def get_best_action(w, state):
    action_values = get_action_values(state, w)
    return ACTIONS[np.argmax(action_values)]

def generate_episode(w, epsilon):
    episode = []
    state = get_start_state()
    action = get_next_action(w, state, epsilon)
    while state[0] < TERMINAL_X:
        next_state, reward = get_next_state_and_reward(state, action)
        episode.append((state, action, reward))
        state = next_state
        action = get_next_action(w, state, epsilon)
    episode.append((state, None, 0))
    return episode

def train_semi_gradient_sarsa(alpha, epsilon, gamma, num_episodes):
    w = np.zeros(NUM_ACTION_FEATURES)
    for episode in range(num_episodes):
        state = get_start_state()
        action = get_next_action(w, state, epsilon)
        while True:
            next_state, reward = get_next_state_and_reward(state, action)
            if next_state[0] >= TERMINAL_X:
                delta = reward - get_action_value(state, action, w)
                w += alpha * delta * get_action_feature_vector(state, action)
                break
            next_action = get_next_action(w, next_state, epsilon)
            delta = reward + gamma * get_action_value(next_state, next_action, w) - get_action_value(state, action, w)
            w += alpha * delta * get_action_feature_vector(state, action)
            state = next_state
            action = next_action
    return w

w = train_semi_gradient_sarsa(alpha=0.5/8, epsilon=0, gamma=1, num_episodes=500)

print(w)

episode = generate_episode(w, epsilon=0)
print(len(episode))
print(episode)