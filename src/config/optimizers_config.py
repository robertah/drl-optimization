import os


class GeneticAlgorithmConfig:
    def __init__(self, config, models_path):
        ga = config['genetic_algorithm']
        self.best = ga['best']
        self.elite = ga['elite']
        self.noise_prob = ga['noise_prob']
        self.models_path = os.path.join(models_path, ga['path'])


class CMAEvolutionStrategiesConfig:
    def __init__(self, config, models_path):
        cma_es = config['cma_evolution_strategies']
        self.perc_selected = cma_es['perc_selected']
        self.models_path = os.path.join(models_path, cma_es['path'])


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
        self.models_path = os.path.join(models_path, ddpg['path'])


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
        self.save_every = td3['save_results_every']


