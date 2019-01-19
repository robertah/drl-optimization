import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """
    Replay Buffer class for transition storage
    """

    def __init__(self, buffer_size):
        """
        Initialize replay buffer

        :param buffer_size: max buffer size
        """
        self.buffer = deque()
        # self.buffer = []
        self.buffer_size = buffer_size
        self.n_experiences = 0

    def get_batch(self, batch_size):
        """
        Sample batch of transitions

        :param batch_size: batch size
        :return: sampled batch
        """
        size = min(batch_size, self.n_experiences)
        batch = random.sample(self.buffer, size)
        return np.asarray([e[0] for e in batch]), np.asarray([e[1] for e in batch]), np.asarray(
            [e[2] for e in batch]), np.asarray([e[3] for e in batch]), np.asarray([e[4] for e in batch])

    def add(self, state, action, reward, new_state, done):
        """
        Add transition to replay buffer

        :param state: state
        :param action: action
        :param reward: reward
        :param new_state: new state
        :param done: done
        :return:
        """
        experience = (state, action, reward, new_state, done)
        if self.n_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.n_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
