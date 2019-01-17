import random
import numpy as np

from optimizers import EvolutionaryOptimizers


class GeneticAlgorithm(EvolutionaryOptimizers):

    def __init__(self, best, elite, noise_prob=0.7):
        """

        :param best: number of best agents to be picked from previous generation as parents for the next one
        :type best: int
        :param elite: number of best agents to keep in the next generation from the previous one
        :type elite: int
        :param noise_prob: probability of sample uniform noise in range [0,0.1] for fine tuning, 1-noise_prob
                probability of sample uniform noise in [0,1] for exploration
        :type noise_prob: float
        """

        self.best = best
        self.elite = elite
        self.noise_prob = noise_prob
        super(GeneticAlgorithm, self).__init__()

    def generate_next_generation(self, population, generation):
        """

        :param population:
        :param generation:
        :return:
        """
        agents = population.agents_weights[generation]
        rewards = population.scores[generation]

        scaled_rewards = np.interp(rewards, (rewards.min(), rewards.max()), (0, 1))
        ordered_indexes = np.argsort(-scaled_rewards)
        best_agents = [agents[i] for i in ordered_indexes[:self.best]] + [
            agents[random.choice(ordered_indexes[self.best:])]]
        new_agents = [agents[i] for i in ordered_indexes[self.elite:]]
        num_layers = len(agents[0])

        next_generation = generation + 1

        for n, agent in enumerate(new_agents):
            child_model = []
            parents = random.sample(range(self.best), 2)

            for i in range(num_layers):
                layer_shape = agent[i].shape
                mother_father = np.random.choice([0, 1], layer_shape)
                new_layer = np.multiply(best_agents[parents[0]][i], mother_father) + np.multiply(
                    best_agents[parents[1]][i], np.ones(layer_shape) - mother_father)
                noise = np.multiply(new_layer, np.random.choice([-1, 0, 1], new_layer.shape)) * \
                        random.uniform(0, np.random.choice([0.01, 0.1], p=[self.noise_prob, 1 - self.noise_prob]))
                new_layer = new_layer + noise
                child_model.append(new_layer)
            population.agents_weights[next_generation][n] = child_model

        for j, b in enumerate(best_agents[:self.elite]):
            population.agents_weights[next_generation][n + j + 1] = b

