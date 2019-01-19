import numpy as np

from config import ENVIRONMENT as env_cfg
from config import TD3_Config as td3_cfg
from .models import Actor, Critic
from .replay_buffer import ReplayBuffer


class TD3:
    """
    Paper: https://arxiv.org/pdf/1802.09477.pdf
    """

    def __init__(self, sess):
        """
        Initialize TD3 class

        :param sess: tensorflow session
        """

        self.env = env_cfg.env
        self.state_size = env_cfg.state_size
        self.action_size = env_cfg.action_size
        self.action_high = self.env.action_space.high
        self.action_low = self.env.action_space.low
        self.batch_size = td3_cfg.batch_size
        self.buffer_size = td3_cfg.buffer_size
        self.buffer_size_warmup = td3_cfg.buffer_size_warmup
        self.tau = td3_cfg.tau
        self.gamma = td3_cfg.gamma
        self.sigma = td3_cfg.sigma
        self.noise_clip = td3_cfg.noise_clip
        self.train_interval = td3_cfg.train_interval
        self.actor_lr = td3_cfg.actor_lr
        self.critic_lr = td3_cfg.critic_lr
        self.actor = Actor(sess, self.state_size, self.action_size, self.action_high, self.action_low, self.actor_lr,
                           self.tau, self.batch_size)
        self.critic1 = Critic(sess, self.state_size, self.action_size, self.critic_lr, self.tau, self.gamma, 'critic1')
        self.critic2 = Critic(sess, self.state_size, self.action_size, self.critic_lr, self.tau, self.gamma, 'critic2')
        self.replay_buffer = ReplayBuffer(self.buffer_size)

    def initialize(self):
        """
        Initialize target networks
        """
        self.update_targets()

    def update_targets(self):
        """
        Update actor, critic 1 and 2 target networks
        """
        self.actor.update_target_network()
        self.critic1.update_target_network()
        self.critic2.update_target_network()

    def get_random_action(self):
        """
        Sample random action from the environment

        :return: sampled action
        """
        return self.env.action_space.sample()

    def get_action(self, state):
        """
        Get action from the actor

        :param state: current state
        :return: action
        """
        return self.actor.get_action(state)

    def get_noisy_action(self, state):
        """
        Get noisy action given the observation

        :param state: current state
        :return: noisy action
        """
        if self.replay_buffer.size > self.buffer_size_warmup:
            action = self.get_action(state)
        else:
            action = self.get_random_action()
        noise = np.clip(np.random.randn(self.action_size) * self.sigma, -self.noise_clip, self.noise_clip)
        noisy_action = action + noise
        return np.clip(noisy_action, self.action_low, self.action_high), action, noise

    def store_experience(self, s, a, r, d, s2):
        """
        Save transition in replay buffer

        :param s: state
        :param a: action
        :param r: reward
        :param d: done
        :param s2: new state
        """
        self.replay_buffer.add(s, a, r, d, s2)

    def train(self, global_step):
        """
        One step policy training

        :param global_step: current global step
        :return: weights if updated, else None
        """

        if self.replay_buffer.size > self.buffer_size_warmup:
            s0, a, r, t, s1 = self.replay_buffer.sample_batch(self.batch_size)
            epsilon = np.clip(np.random.randn(self.batch_size, self.action_size), -self.noise_clip, self.noise_clip)
            target_actions = self.actor.get_target_action(s1) + epsilon
            target_actions = np.clip(target_actions, self.action_low, self.action_high)
            target_qval = self.get_target_qval(s1, target_actions)
            t = t.astype(dtype=int)
            y = r + self.gamma * target_qval * (1 - t)
            self.critic1.train(s0, a, y)
            self.critic2.train(s0, a, y)
            if global_step % self.train_interval == 0:
                actions = self.actor.get_action(s0)
                grads = self.critic1.get_action_gradients(s0, actions)
                w = self.actor.train(s0, grads[0])
                self.update_targets()
                return w
        return None

    def get_target_qval(self, state, action):
        """
        Get target Q value

        :param state: state
        :param action: action
        :return:
        """

        target_qval1 = self.critic1.get_target_qval(state, action)
        target_qval2 = self.critic2.get_target_qval(state, action)
        return np.minimum(target_qval1, target_qval2)

    def get_qval(self, state, action):
        """
        Get Q value from critics

        :param state: state
        :param action: action
        :return: action
        """
        qval1 = self.critic1.get_qval(state, action)
        qval2 = self.critic2.get_qval(state, action)
        return np.minimum(qval1, qval2)
