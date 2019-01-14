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
        self.environment = environment

        self.actor = Actor(state_size=self.environment.state_size, action_size=self.environment.action_size,
                           learning_rate=LEARNING_RATE)
        self.critic = Critic(self.agent.env.state_size, self.environment.action_size, LEARNING_RATE)

        self.target_actor = copy(self.actor)
        self.target_critic = copy(self.critic)

        self.exploration_noise = OrnsteinUhlenbeckNoise(self.environment.action_size)

        self.replay_buffer = ReplayBuffer(MAX_BUFFER)

    def get_exploitation_action(self, state):
        action = self.target_actor.predict(state)
        return action

    def get_exploration_action(self, state):
        action = self.actor.predict(state)
        new_action = action + self.exploration_noise.noise()
        return new_action

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """

        s1, a1, r1, s2 = self.replay_buffer.sample_batch(BATCH_SIZE)

        self.critic.optimize(self.target_actor, self.target_critic, s1, a1, r1, s2)
        self.actor.optimize(self.critic, s1)

        # utils.soft_update(self.target_actor, self.actor, TAU)
        # utils.soft_update(self.target_critic, self.critic, TAU)
        self.target_actor = update_target(self.target_actor, self.actor, TAU)
        self.target_critic = update_target(self.target_critic, self.critic, TAU)

    def run(self):
        for e in range(MAX_EPISODES):
            state = self.environment.env.reset()
            for i in range(self.environment.max_time):
                action = self.get_exploration_action(state)
                new_state, reward, done, info = self.environment.env.step(action)
                if done:
                    new_state = None
                else:
                    self.replay_buffer.add(state, action, reward, new_state, done)

                state = new_state
                self.optimize()

                if done:
                    break

