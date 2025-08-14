import autograd.numpy as np
from autograd import grad

states = [0, 1, 2, 3, 4]
interest = [1, 0, 0, 0, 0]
start_state = 0
terminal_states = [4]

TRUE_VALUES = [4, 3, 2, 1, 0]

def get_reward_and_next_state(state, action='right'):
    if action == 'right':
        if state == 4:
            return 0, 4
        else:
            return 1, state + 1
    elif action == 'left':
        if state == 0:
            return 0, 0
        else:
            return 1, state - 1
    else:
        raise ValueError(f"Invalid action: {action}")

def feature_vector(state):
    if state == 4:
        return np.array([0.0, 0.0], dtype=np.float64)
    if state == 0 or state == 1:
        return np.array([1.0, 0.0], dtype=np.float64)
    if state == 2 or state == 3:
        return np.array([0.0, 1.0], dtype=np.float64)
    raise ValueError(f"Invalid state: {state}")

def value_function(w, state):
    if state in terminal_states:
        return 0.0
    return np.dot(w, feature_vector(state))

def generate_episode():
    state = start_state
    episode = []
    while state not in terminal_states:
        reward, next_state = get_reward_and_next_state(state, 'right')
        episode.append((state, reward))
        state = next_state
    return episode

def loss(w, state, target):
    v_hat = value_function(w, state)
    return 0.5 * (target - v_hat) ** 2
    
grad_loss = grad(loss)

def gradient_mc_update(w, num_episodes, alpha, gamma=1.0):
    for _ in range(num_episodes):
        episode = generate_episode()  # list of (state, reward)
        G = 0.0
        M = 0.0
        for i, (state, reward) in enumerate(reversed(episode)):
            G = reward + gamma * G
            g = grad_loss(w, state, G)
            w -= alpha * interest[state] * g
    return w


w = np.zeros(2, dtype=np.float64)
w = gradient_mc_update(w, 10000, 0.001)
print(w)
estimates = [value_function(w, s) for s in states]
print(estimates)