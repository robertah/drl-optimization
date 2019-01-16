# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of a TD3 agent.

Implementation of TD3 - Twin Delayed Deep Deterministic Policy Gradient
Algorithm and hyperparameter details can be found here:
https://arxiv.org/pdf/1802.09477.pdf

"""

from .replay_buffer import ReplayBuffer
from .models import ActorNetwork, CriticNetwork
import numpy as np

LR_C = 2e-4
LR_A = 1e-4
GAMMA = 0.99
TAU = 0.001
NOISE_STD = 0.1
BUFFER_SIZE = 1000000
WARMUP_BUFFER = 10000
BATCH_SIZE = 32
N_EPISODES = 3000
TEST_STEPS = 100

class TD3():
    """TD3 agent."""

    def __init__(self, env, sess):
        """Initialize members."""
        state_dim = env.observation_space.shape[0]
        self.env = env
        self.action_dim = env.action_space.shape[0]
        self.action_high = env.action_space.high
        self.action_low = env.action_space.low
        self.batch_size = BATCH_SIZE
        self.warmup_size = WARMUP_BUFFER
        self.gamma = GAMMA
        self.sigma = 0.5
        self.sigma_tilda = 0.1
        self.noise_cap = 0.1
        self.train_interval = 2
        self.actor = ActorNetwork(sess=sess,
                                  state_dim=state_dim,
                                  action_dim=self.action_dim,
                                  action_high=self.action_high,
                                  action_low=self.action_low,
                                  learning_rate=LR_A,
                                  tau=TAU,
                                  batch_size=BATCH_SIZE)
        self.critic1 = CriticNetwork(sess=sess,
                                     state_dim=state_dim,
                                     action_dim=self.action_dim,
                                     learning_rate=LR_C,
                                     tau=TAU,
                                     gamma=GAMMA,
                                     name='critic1')
        self.critic2 = CriticNetwork(sess=sess,
                                     state_dim=state_dim,
                                     action_dim=self.action_dim,
                                     learning_rate=LR_C,
                                     tau=TAU,
                                     gamma=GAMMA,
                                     name='critic2')
        self.replay_buffer = ReplayBuffer(buffer_size=BUFFER_SIZE)

    def initialize(self):
        """Initialization before playing."""
        self.update_targets()

    def random_action(self, observation):
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
            action = self.random_action(observation)
        noise = np.clip(np.random.randn(self.action_dim) * self.sigma,
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
            epsilon = np.clip(np.random.randn(self.batch_size, self.action_dim),
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
