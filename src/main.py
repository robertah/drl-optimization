import os

from genetic import Population

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# agents_weights, scores, children = run_agent_genetic(n_agents=50, n_generations=20, save=True)


# weights = get_best_agent(mean=True, timestamp='20190103183514')
# perturbate_weights(weights)


agents = Population(population_size=50,  # n agents
                    max_generations=100,  # max n generations
                    n_selected=0.3,  # n agents selected for crossover
                    mutation_rate=0.3,  # probability of mutation
                    mutation_noise=1,  # gaussian noise scale for mutation
                    elite=2  # n best agents kept for next generation
                    )
agents.evolve()
