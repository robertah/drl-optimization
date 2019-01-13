class GeneticAlgorithmConfig:
    def __init__(self, config):
        ga = config['genetic_algorithm']
        self.selected = ga['perc_selected']
        self.mutation_rate = ga['mutation_rate']
        self.mutation_noise = ga['mutation_noise']
        self.elite = ga['elite']


class CMAEvolutionStrategiesConfig:
    def __init__(self, config):
        cma_es = config['cma_evolution_strategies']
        self.selected = cma_es['perc_selected']


class EvolutionStrategiesConfig:
    def __init__(self, config):
        es = config['evolution_strategies']
        self.learning_rate = es['learning_rate']
        self.noise = es['noise']


class DDPGConfig:
    def __init__(self, config):
        ddpg = config['ddpg']
        self.replay_start = ddpg['replay_start']
        self.buffer_size = ddpg['buffer_size']
        self.batch_size = ddpg['batch_size']
        self.gamma = ddpg['gamma']
        self.tau = ddpg['tau']
        self.actor_lr = ddpg['actor_lr']
        self.critic_lr = ddpg['critic_lr']
        self.n_episodes = ddpg['n_episodes']
        self.n_runs = ddpg['n_runs']
