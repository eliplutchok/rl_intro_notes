import matplotlib.pyplot as plt
import autograd.numpy as np
import numpy as onp
from autograd import grad

states = np.arange(0, 3)
actions = ['left', 'right']

def feature_vector(s, a):
    if a == 'left':
        return np.array([0, 1])
    else:
        return np.array([1, 0])

def get_next_state_and_reward(s, a):
    direction = 1 if a == 'right' else -1
    if s == 1:
        direction *= -1
    next_state = s + direction
    if next_state < 0:
        next_state = 0
    return next_state, -1

def pi(s, a, theta):
    return np.exp(np.dot(theta, feature_vector(s, a))) / np.sum([np.exp(np.dot(theta, feature_vector(s, act))) for act in actions])

def generate_episode(theta, max_steps=1000):
    episode = []
    state = 0
    while state != 3 and len(episode) < max_steps:
        action = onp.random.choice(actions, p=[pi(state, a, theta) for a in actions])
        next_state, reward = get_next_state_and_reward(state, action)
        episode.append((state, action, reward))
        state = next_state
    return episode

def reinforce(policy, alpha = 2 ** -12, gamma = 1.0, num_episodes = 1000):
    returns = []

    W = np.zeros(2)
    for _ in range(num_episodes):
        frozen_W = W.copy()
        episode = generate_episode(frozen_W)
        G = 0
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            logpi = lambda theta: np.log(policy(state, action, theta))
            grad_logpi = grad(logpi)(frozen_W)
            W += alpha * G * grad_logpi
        returns.append(G)
    return W, returns

def get_100_run_return_avg(alpha):
    all_run_returns = []
    for _ in range(100):
        _, episode_returns = reinforce(pi, alpha = alpha)
        all_run_returns.append(onp.asarray(episode_returns))
    return onp.mean(onp.stack(all_run_returns, axis=0), axis=0)

returns_12 = get_100_run_return_avg(2 ** -12)
returns_13 = get_100_run_return_avg(2 ** -13)
returns_14 = get_100_run_return_avg(2 ** -14)

plt.plot(returns_12, label='alpha = 2 ** -12')
plt.plot(returns_13, label='alpha = 2 ** -13')
plt.plot(returns_14, label='alpha = 2 ** -14')
plt.axhline(y=-11, color='black', linestyle='--')
plt.ylim(-100, 0)
plt.legend()
plt.show()
