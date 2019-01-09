class PopulationConfig:

    def __init__(self, config):
        population = config['population']
        self.size = population['size']
        self.max_generations = population['max_generations']
