import os

from genetic import Population
from config import GA

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# agents_weights, scores, children = run_agent_genetic(n_agents=50, n_generations=20, save=True)


# weights = get_best_agent(mean=True, timestamp='20190103183514')
# perturbate_weights(weights)


agents = Population(population_size=GA.population_size,  # n agents
                    max_generations=GA.max_generations,  # max n generations
                    n_selected=GA.selected,  # n agents selected for crossover
                    mutation_rate=GA.mutation_rate,  # probability of mutation
                    mutation_noise=GA.mutation_noise,  # gaussian noise scale for mutation
                    elite=GA.elite  # n best agents kept for next generation
                    )
agents.evolve()
