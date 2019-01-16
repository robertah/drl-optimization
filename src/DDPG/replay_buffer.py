import random
from collections import deque

import numpy as np


class ReplayBuffer:

    def __init__(self, buffer_size):
        self.buffer = deque()
        # self.buffer = []
        self.buffer_size = buffer_size
        self.n_experiences = 0

    def get_batch(self, batch_size):
        size = min(batch_size, self.n_experiences)
        batch = random.sample(self.buffer, size)
        return np.asarray([e[0] for e in batch]), np.asarray([e[1] for e in batch]), np.asarray(
            [e[2] for e in batch]), np.asarray([e[3] for e in batch]), np.asarray([e[4] for e in batch])

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.n_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.n_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    # def sort_by_td(self):
    #    self.buffer = sorted(self.buffer, key=lambda x: x[-1])
