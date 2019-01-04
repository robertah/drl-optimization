from genetic import run_agent_genetic, Agent
from utils import get_results, get_best_agent
from analysis import perturbate_weights

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# agents_weights, scores, children = run_agent_genetic(n_agents=50, n_generations=20, save=True)


weights = get_best_agent(mean=True, timestamp='20190103183514')
perturbate_weights(weights)
