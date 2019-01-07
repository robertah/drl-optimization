import os
import yaml
from .environment_config import EnvironmentConfig
from .genetic_algorithm_config import GeneticAlgorithmConfig

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..'))

with open(os.path.join(ROOT_DIR, 'config.yml')) as c:
    config = yaml.load(c)

with open(os.path.join(ROOT_DIR, 'src', 'config', 'models.yml')) as m:
    models = yaml.load(m)

RANDOM_SEED = config['random_seed']
ENVIRONMENT = EnvironmentConfig(config, models, RANDOM_SEED)

GA = GeneticAlgorithmConfig(config)

RESULTS_SCORES = os.path.join(ROOT_DIR, config['results']['path'], config['results']['scores'])
RESULTS_WEIGHTS = os.path.join(ROOT_DIR, config['results']['path'], config['results']['weights'])

VISUALIZATION_WEIGHTS = os.path.join(ROOT_DIR, config['visualization']['path'], config['visualization']['weights'])

LOGGER = config['logger']