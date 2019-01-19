import numpy as np

class OrnsteinUhlenbeckNoise:
    """
    Ornstein Uhlenbeck process for noise generation
    """
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        """
        Initialize Ornstein Uhlenbeck process

        :param action_space: gym environment's action space
        :param mu:
        :param theta:
        :param max_sigma:
        :param min_sigma:
        :param decay_period:
        """
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
        """
        Reset state
        """
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        """
        Generate noise
        :return:
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_noisy_action(self, action, t=0):
        """
        Generate noisy action

        :param action:
        :param t:
        :return:
        """
        ou_state = self.noise()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
