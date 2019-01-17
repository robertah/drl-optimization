import os


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
    def __init__(self, config, models_path):
        ddpg = config['ddpg']
        self.buffer_size = ddpg['buffer_size']
        self.batch_size = ddpg['batch_size']
        self.gamma = ddpg['gamma']
        self.tau = ddpg['tau']
        self.actor_lr = ddpg['actor_lr']
        self.critic_lr = ddpg['critic_lr']
        self.n_episodes = ddpg['n_episodes']
        self.actor_weights = os.path.join(models_path, ddpg['saved_weights']['actor'])
        self.critic_weights = os.path.join(models_path, ddpg['saved_weights']['critic'])
        self.target_actor_weights = os.path.join(models_path, ddpg['saved_weights']['target_actor'])
        self.target_critic_weights = os.path.join(models_path, ddpg['saved_weights']['target_critic'])


class TD3Config:
    def __init__(self, config, models_path):
        td3 = config['td3']
        self.n_episodes = td3['n_episodes']
        self.batch_size = td3['batch_size']
        self.buffer_size = td3['buffer_size']
        self.buffer_size_warmup = td3['buffer_size_warmup']
        self.gamma = td3['gamma']
        self.tau = td3['tau']
        self.sigma = td3['sigma']
        self.sigma_tilda = td3['sigma_tilda']
        self.noise_cap = td3['noise_cap']
        self.train_interval = td3['train_interval']
        self.test_every = td3['test_every']
        self.record_videos = td3['record_videos']
        self.actor_lr = td3['actor_lr']
        self.critic_lr = td3['critic_lr']
        self.models_path = os.path.join(models_path, td3['path'])


