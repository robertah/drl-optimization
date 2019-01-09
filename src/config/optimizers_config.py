class GeneticAlgorithmConfig:
    def __init__(self, config):
        ga = config['genetic_algorithm']
        self.selected = ga['perc_selected']
        self.mutation_rate = ga['mutation_rate']
        self.mutation_noise = ga['mutation_noise']
        self.elite = ga['elite']


class CMAEvolutionStrategiesConfig:
    def __init__(self, config):
        cma_es = config['cma_evolution_strategies']
        self.selected = cma_es['perc_selected']
