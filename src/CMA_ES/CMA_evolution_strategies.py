import numpy as np
from optimizers import EvolutionaryOptimizers
from config import CMAEvolutionStrategiesConfig


class CMAEvolutionStrategies(EvolutionaryOptimizers):
    """
    Simplified Covariance Matrix Adaptation Evolutionary Strategies
    """

    def __init__(self):
        self.perc_selected = CMAEvolutionStrategiesConfig.perc_selected
        super(CMAEvolutionStrategies, self).__init__()

    def selection(self, scores, population_size):
        """
        Select the best performing agents in the previous generation as new parents

        :param scores: agents' score
        :param population_size: population size (n agents per generation)

        :return: indices of the best performing agents
        """
        n_selected = int(self.perc_selected * population_size)
        selected_parents = np.argsort(-scores)[:n_selected]
        return selected_parents

    @staticmethod
    def compute_covariance(parents):
        """
        Compute covariance of weights

        :param parents: selected parents for breeding
        :return: parents' covariance
        """
        parents -= np.mean(parents, axis=0)
        covariance = np.dot(parents.T, parents.conj()) / (len(parents) - 1)
        return covariance

    def sample_children(self, population, parents):
        """
        Sample children from Gaussian distribution with parents' mean and covariance

        :param population: population of agents
        :param parents: selected parents
        :return: sampled children
        """
        weights_size = len(population.agents_weights[0][0])
        sampled = np.empty(weights_size, dtype=np.ndarray)
        for j in range(weights_size):
            temp = parents[:, j]
            orig_shape = temp[0].shape
            parents_flatten = np.empty((temp.shape[0], np.prod(temp[0].shape)), dtype=float)
            for i, p in enumerate(temp):
                parents_flatten[i] = p.ravel()

            # scaler = StandardScaler()
            # parents_scaled = scaler.fit_transform(parents_flatten)
            mean = np.mean(parents_flatten, axis=0)
            covariance = self.compute_covariance(parents_flatten)
            children = np.random.multivariate_normal(mean, covariance, size=population.size)
            if len(orig_shape) == 1:
                reshape = (population.size, orig_shape[0])
            else:
                reshape = (population.size, orig_shape[0], orig_shape[1])
            sampled[j] = children.reshape(reshape)

        return sampled

    def generate_next_generation(self, population, generation):
        """
        Update population's next generation

        :param population: population of agents
        :param generation: current generation id
        """
        # get ids of the best performing agents
        id = self.selection(scores=population.scores[generation], population_size=population.size)
        sampled = self.sample_children(population, population.agents_weights[generation][id])
        next_generation = generation + 1
        for i in range(population.size):
            child = []
            for j in range(sampled.shape[0]):
                child.append(sampled[j][i])
            population.agents_weights[next_generation][i] = child

