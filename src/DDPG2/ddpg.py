from copy import copy
from keras.losses import mean_squared_error
import numpy as np
from .model import Actor, Critic, update_target
from .noise import OrnsteinUhlenbeckNoise
from .replay_buffer import ReplayBuffer

MAX_BUFFER = 1000000
BATCH_SIZE = 128
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001

MAX_EPISODES = 5000
MAX_STEPS = 1000
MAX_BUFFER = 1000000


class DDPG:

    def __init__(self, environment):
        """
        Initialize Deep Deterministic Policy Gradients

        :param environment: environment configuration
        :type environment: config.EnvironmentConfig
        """
        self.environment = environment

        # initialize actor and critic networks
        self.actor = Actor(self.environment.state_size, self.environment.action_size, LEARNING_RATE)
        self.critic = Critic(self.environment.state_size, self.environment.action_size, LEARNING_RATE)

        # initialize target networks
        self.target_actor = copy(self.actor)
        self.target_critic = copy(self.critic)

        # initialize Ornstein Uhlenbeck process
        self.exploration_noise = OrnsteinUhlenbeckNoise(self.environment.action_size)

        # initialize replay buffer to store transitions
        self.replay_buffer = ReplayBuffer(MAX_BUFFER)

    def get_exploitation_action(self, state):
        """
        Get action

        :param state: current state
        :return: action
        """
        action = self.target_actor.predict(state)
        return action

    def get_exploration_action(self, state):
        """
        Get noisy action

        :param state: current state
        :return: action
        """
        action = self.actor.predict(state) + self.exploration_noise.noise()
        return action

    def optimize(self):
        """
        Optimize actor and critic network. Update target actor and critic's weights
        """

        # sample a random minibatch of transitions from replay buffer
        s1, a1, r1, s2 = self.replay_buffer.sample_batch(BATCH_SIZE)

        # optimize actor and critic
        self.critic.optimize(self.target_actor, self.target_critic, s1, a1, r1, s2)
        self.actor.optimize(self.critic, s1)

        # update target actor and target critic
        self.target_actor = update_target(self.target_actor, self.actor, TAU)
        self.target_critic = update_target(self.target_critic, self.critic, TAU)

    def run(self):
        for e in range(MAX_EPISODES):

            # get initial observation state
            state = self.environment.env.reset()

            for i in range(self.environment.max_time):

                # select action according to current policy and exploration noise
                action = self.get_exploration_action(state)

                # execute action
                new_state, reward, done, info = self.environment.env.step(action)

                if done:
                    new_state = None
                else:
                    # store transition in replay buffer
                    self.replay_buffer.add(state, action, reward, new_state, done)

                state = new_state

                # optimize actor and critic, and update target networks
                self.optimize()

                if done:
                    break

