class GeneticAlgorithmConfig:
    def __init__(self, config):
        ga = config['genetic_algorithm']
        self.population_size = ga['population_size']
        self.max_generations = ga['max_generations']
        self.selected = ga['n_selected']
        self.mutation_rate = ga['mutation_rate']
        self.mutation_noise = ga['mutation_noise']
        self.elite = ga['elite']
