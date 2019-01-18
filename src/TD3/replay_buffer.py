from collections import deque
import numpy as np
import random


class ReplayBuffer:

    def __init__(self, buffer_size):
        """
        Initialize replay buffer for transitions storage

        :param buffer_size: max_size of the replay buffer
        """

        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, d, s2):
        """
        Add (state, action, reward, done, new_state) transition to the buffer

        :param s: state
        :param a: action
        :param r: reward
        :param d: done
        :param s2: new state
        """
        transition = (s, a, r, d, s2)
        if self.count < self.buffer_size:
            self.buffer.append(transition)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(transition)

    @property
    def size(self):
        """
        Get current replay buffer size
        :return: current replay buffer size
        """
        return self.count

    def sample_batch(self, batch_size):
        """
        Sample a batch of transitions from replay buffer

        :param batch_size:
        :return: sampled batch
        """
        
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([i[0] for i in batch])
        a_batch = np.array([i[1] for i in batch])
        r_batch = np.array([i[2] for i in batch])
        t_batch = np.array([i[3] for i in batch])
        s2_batch = np.array([i[4] for i in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch
