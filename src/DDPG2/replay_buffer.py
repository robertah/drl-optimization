from collections import deque
import random


class ReplayBuffer:
    """
    Replay buffer with finite sized cache. Transitions are sampled from the environment according to the exploration
    policy and the tuple (s_t, a_t, r_t, s_t+1) is stored in the replay buffer. When the replay buffer is full the
    oldest samples are discarded.
    """

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.n_experiences = 0
        self.buffer = deque(maxlen=self.buffer_size)

    def sample_batch(self, batch_size):
        """
        Sample a batch of stored transitions

        :param batch_size: size of the batch
        :return:
        """
        if self.n_experiences < batch_size:
            return random.sample(self.buffer, self.n_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def add(self, state, action, reward, new_state):
        """
        Add new transition (s_t, a_t, r_t, s_t+1)

        :param state:
        :param action:
        :param reward:
        :param new_state:
        """
        experience = (state, action, reward, new_state)
        if self.n_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.n_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
