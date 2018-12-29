from genetic import run_agent_genetic, Agent
from utils import get_results, get_best_agent

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# agents_weights, scores, children = run_agent_genetic(n_agents=50, n_generations=50, save=True)

weights, scores = get_results()

best_weights = get_best_agent()

agent = Agent(weights=best_weights)
agent.run_agent(render=True)