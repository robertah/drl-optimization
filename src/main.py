#from genetic.agent import Agent
#from utils import get_results, get_best_agent
import sys
import os

from genetic import run_agent_genetic_positive, run_genetic_no_mean_parallel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
scores = run_genetic_no_mean_parallel(n_agents=100, n_generations=300, save=True)

#weights, scores = get_results()

#best_weights = get_best_agent()

#agent = Agent(weights=best_weights)
#agent.run_agent(render=True)