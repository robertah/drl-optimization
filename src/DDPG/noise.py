import numpy as np


# class OrnsteinUhlenbeckNoise:
#
#     def __init__(self, action_size, mu=0, theta=0.15, sigma=0.2):
#         self.action_size = action_size
#         self.mu = mu
#         self.theta = theta
#         self.sigma = sigma
#         self.state = np.ones(self.action_size) * self.mu
#         self.reset()
#
#     def reset(self):
#         self.state = np.ones(self.action_size) * self.mu
#
#     def noise(self):
#         x = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
#         self.state = x + dx
#         return self.state
#
#     def get_noisy_action(self, action):
#         return action + self.noise()


class OrnsteinUhlenbeckNoise:
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_noisy_action(self, action, t=0):
        ou_state = self.noise()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
