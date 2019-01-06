import gym

class EnvironmentConfig:
    def __init__(self, config, models):
        environment = config['environment']
        self.name = environment['name']
        self.env = gym.make(self.name)
        self.max_time = environment['max_time']
        self.animate = environment['animate']
        for e in models['environments']:
            if e['name'] == self.name:
                self.hidden_units = e['n_hidden_units']
