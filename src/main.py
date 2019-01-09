import os

from config import POPULATION, GA, CMA_ES
from genetic import GeneticAlgorithm
from CMA_ES import CMAEvolutionStrategies
from population import Population

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# agents_weights, scores, children = run_agent_genetic(n_agents=50, n_generations=20, save=True)


# weights = get_best_agent(mean=True, timestamp='20190103183514')
# perturbate_weights(weights)

ga = GeneticAlgorithm(perc_selected=GA.selected, mutation_rate=GA.mutation_rate,
                      mutation_noise=GA.mutation_noise, elite=GA.elite)

cma_es = CMAEvolutionStrategies(perc_selelcted=CMA_ES.selected)

agents = Population(size=POPULATION.size, max_generations=POPULATION.max_generations, optimizer=cma_es)

agents.evolve()

# weights = utils.get_best_agent(mean=False)
# agent = Agent(ENVIRONMENT, weights)
# agent.run_agent(render=True)
# perturbate_weights(weights)
