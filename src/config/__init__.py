import os
import yaml
from .environment_config import EnvironmentConfig

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..'))

with open(os.path.join(ROOT_DIR, 'config.yml')) as c:
    config = yaml.load(c)

with open(os.path.join(ROOT_DIR, 'src', 'config', 'models.yml')) as m:
    models = yaml.load(m)

RANDOM_SEED = config['random_seed']
ENVIRONMENT = EnvironmentConfig(config, models)

GA_MUTATION_NOISE = config['genetic_algorithm']['mutation_noise']

RESULTS_SCORES = os.path.join(ROOT_DIR, config['results']['path'], config['results']['scores'])
