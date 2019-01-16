from collections import deque
import numpy as np
import random
import copy


class ReplayBuffer(object):
    """Replay buffer."""

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, t, s2):
        """Add experience to the buffer.

        Add experience (s, a, r, t, s2) to the buffer.

        Args:
          s  - state/observation at time t
          a  - action taken at time t
          r  - reward received from the environment
          t  - termination flag, indicating whether the episode ends
          s2 - state/observation at time t+1
        """
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    @property
    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        """Sample a batch of experience."""
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch
