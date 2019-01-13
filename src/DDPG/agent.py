import numpy as np
from .noise import OrnsteinUhlenbeckNoise as Noise


class Agent:

    def __init__(self, environment_config, weights=None):
        self.env = environment_config

    def get_action(self, state, actor, noise=False):
        action = actor.action(state)
        if noise:
            action += Noise(self.env.action_size).noise()
        return action

