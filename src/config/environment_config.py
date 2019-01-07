import gym


class EnvironmentConfig:
    def __init__(self, config, models, random_seed):
        environment = config['environment']
        self.name = environment['name']
        self.env = gym.make(self.name)
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            self.state_size = self.env.observation_space.n
        else:
            self.state_size = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_size = self.env.action_space.n
        else:
            self.action_size = self.env.action_space.shape[0]
        if random_seed:
            self.env.seed(random_seed)
        self.n_runs = environment['n_runs_per_agent']
        self.max_time = environment['max_time']
        self.animate = environment['animate']
        for e in models['environments']:
            if e['name'] == self.name:
                if len(e['n_hidden_units']) != (len(e['activations']) - 1):
                    raise ValueError("List of hidden units and list of activation functions are not consistent.")
                self.hidden_units = e['n_hidden_units']
                self.activations = e['activations']
