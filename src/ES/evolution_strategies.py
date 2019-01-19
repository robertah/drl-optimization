import numpy as np

from optimizers import EvolutionaryOptimizers
from config import ES


class EvolutionStrategies(EvolutionaryOptimizers):

    def __init__(self):
        self.noise = ES.noise
        super(EvolutionStrategies, self).__init__()

    @staticmethod
    def crossover(agents, rewards):
        """
        :param agents: a list of keras neural networks (parents agents)
        :param rewards: list (or array) of rewards associated to the performance of the
              corresponding model
        :return: the child weights computed by the weighted average
              of the parents w.r.t. the reward
        """
        rewards = np.array(rewards)
        num_layers = len(agents[0])
        normalized_rewards = rewards / np.sum(rewards)
        child_model = []
        for i in range(num_layers):
            new_layer = np.zeros_like(agents[0][i])
            for j, parent_agent in enumerate(agents):
                layer = parent_agent[i] * normalized_rewards[j]
                new_layer = new_layer + layer
            child_model.append(new_layer)
        return child_model

    def generate_next_generation(self, population, generation):
        """
        Update population's next generation

        :param population: population of agents
        :param generation: current generation id
        """

        child = self.crossover(population.agents_weights[generation], population.scores[generation])

        next_generation = generation + 1
        for i in range(population.size):
            new_child = []
            for layer in child:
                new_layer = np.random.normal(layer, self.noise)
                new_child.append(new_layer)

            population.agents_weights[next_generation][i] = new_child
