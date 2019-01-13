# -----------------------------------
# Deep Deterministic Policy Gradient
# Author: Flood Sung
# Date: 2016.5.4
# -----------------------------------
import gym
import tensorflow as tf
import numpy as np
from .noise import OrnsteinUhlenbeckNoise
from .critic import CriticNetwork
from .actor import ActorNetwork
from .replay_buffer import ReplayBuffer

from config import ENVIRONMENT
from .agent import Agent

# Hyper Parameters:

REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99


class DDPG:
    """docstring for DDPG"""

    def __init__(self, agent):
        self.agent = agent
        self.environment = agent.env
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = self.environment.state_size
        self.action_dim = self.environment.action_size

        self.sess = tf.InteractiveSession()

        self.actor_network = ActorNetwork(self.sess, self.state_dim, self.action_dim)
        self.critic_network = CriticNetwork(self.sess, self.state_dim, self.action_dim)

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OrnsteinUhlenbeckNoise(self.action_dim)

    def train(self):
        # print "train step",self.time_step
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch, [BATCH_SIZE, self.action_dim])

        # Calculate y_batch

        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch, next_action_batch)
        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch, [BATCH_SIZE, 1])
        # Update critic by minimizing the loss L
        self.critic_network.train(y_batch, state_batch, action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch, action_batch_for_gradients)

        self.actor_network.train(q_gradient_batch, state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

    def noise_action(self, state):
        # Select action a_t according to the current policy and exploration noise
        action = self.actor_network.action(state)
        return action + self.exploration_noise.noise()

    def action(self, state):
        action = self.actor_network.action(state)
        return action

    def perceive(self, state, action, reward, next_state, done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Store transitions to replay start size then start training
        if self.replay_buffer.n_experiences > REPLAY_START_SIZE:
            self.train()

        # if self.time_step % 10000 == 0:
        # self.actor_network.save_network(self.time_step)
        # self.critic_network.save_network(self.time_step)

        # Re-iniitialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()

    def run(self):

        EPISODES = 500
        TEST = 10

        agent = Agent(ENVIRONMENT)

        for episode in range(EPISODES):
            state = self.environment.env.reset()
            # print "episode:",episode
            # Train
            for step in range(agent.env.max_time):
                action = self.noise_action(state)
                next_state, reward, done, _ = agent.env.env.step(action)
                self.perceive(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
            # Testing:
            if episode % 100 == 0 and episode > 100:
                total_reward = 0
                for i in range(TEST):
                    state = agent.env.env.reset()
                    for j in range(agent.env.max_time):
                        # env.render()
                        action = self.action(state)  # direct action for test
                        state, reward, done, _ = agent.env.env.step(action)
                        total_reward += reward
                        if done:
                            break
                ave_reward = total_reward / TEST
                print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)






