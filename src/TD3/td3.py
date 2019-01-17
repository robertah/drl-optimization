import numpy as np

from config import ENVIRONMENT as env_cfg
from config import TD3_Config as td3_cfg
from .models import Actor, Critic
from .replay_buffer import ReplayBuffer


class TD3():
    """TD3 agent."""

    def __init__(self, sess):
        """Initialize members."""
        self.env = env_cfg.env
        self.state_size = env_cfg.state_size
        self.action_size = env_cfg.action_size
        self.action_high = self.env.action_space.high
        self.action_low = self.env.action_space.low
        self.batch_size = td3_cfg.batch_size
        self.buffer_size = td3_cfg.buffer_size
        self.warmup_size = td3_cfg.buffer_size_warmup
        self.tau = td3_cfg.tau
        self.gamma = td3_cfg.gamma
        self.sigma = td3_cfg.sigma
        self.sigma_tilda = td3_cfg.sigma_tilda
        self.noise_cap = td3_cfg.noise_cap
        self.train_interval = td3_cfg.train_interval
        self.actor_lr = td3_cfg.actor_lr
        self.critic_lr = td3_cfg.critic_lr
        self.actor = Actor(sess, self.state_size, self.action_size, self.action_high, self.action_low, self.actor_lr,
                           self.tau, self.batch_size)
        self.critic1 = Critic(sess, self.state_size, self.action_size, self.critic_lr, self.tau, self.gamma, 'critic1')
        self.critic2 = Critic(sess, self.state_size, self.action_size, self.critic_lr, self.tau, self.gamma, 'critic2')
        self.replay_buffer = ReplayBuffer(self.buffer_size)

    def initialize(self):
        """Initialization before playing."""
        self.update_targets()

    def random_action(self):
        """Return a random action."""
        return self.env.action_space.sample()

    def action(self, observation):
        """Return an action according to the agent's policy."""
        return self.actor.get_action(observation)

    def action_with_noise(self, observation):
        """Return a noisy action."""
        if self.replay_buffer.size > self.warmup_size:
            action = self.action(observation)
        else:
            action = self.random_action()
        noise = np.clip(np.random.randn(self.action_size) * self.sigma,
                        -self.noise_cap, self.noise_cap)
        action_with_noise = action + noise
        return (np.clip(action_with_noise, self.action_low, self.action_high),
                action, noise)

    def store_experience(self, s, a, r, t, s2):
        """Save experience to replay buffer."""
        self.replay_buffer.add(s, a, r, t, s2)

    def train(self, global_step):
        """Train the agent's policy for 1 iteration."""
        if self.replay_buffer.size > self.warmup_size:
            s0, a, r, t, s1 = self.replay_buffer.sample_batch(self.batch_size)
            epsilon = np.clip(np.random.randn(self.batch_size, self.action_size),
                              -self.noise_cap, self.noise_cap)
            target_actions = self.actor.get_target_action(s1) + epsilon
            target_actions = np.clip(target_actions,
                                     self.action_low,
                                     self.action_high)
            target_qval = self.get_target_qval(s1, target_actions)
            t = t.astype(dtype=int)
            y = r + self.gamma * target_qval * (1 - t)
            self.critic1.train(s0, a, y)
            self.critic2.train(s0, a, y)
            if global_step % self.train_interval == 0:
                actions = self.actor.get_action(s0)
                grads = self.critic1.get_action_gradients(s0, actions)
                self.actor.train(s0, grads[0])
                self.update_targets()

    def update_targets(self):
        """Update all target networks."""
        self.actor.update_target_network()
        self.critic1.update_target_network()
        self.critic2.update_target_network()

    def get_target_qval(self, observation, action):
        """Get target Q-val."""
        target_qval1 = self.critic1.get_target_qval(observation, action)
        target_qval2 = self.critic2.get_target_qval(observation, action)
        return np.minimum(target_qval1, target_qval2)

    def get_qval(self, observation, action):
        """Get Q-val."""
        qval1 = self.critic1.get_qval(observation, action)
        qval2 = self.critic2.get_qval(observation, action)
        return np.minimum(qval1, qval2)
