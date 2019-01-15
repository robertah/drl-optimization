import numpy as np

from optimizers import EvolutionaryOptimizers


class GeneticAlgorithm(EvolutionaryOptimizers):

    def __init__(self, perc_selected, mutation_rate, mutation_noise, elite=None):

        if not (0 <= mutation_rate <= 1):
            raise ValueError("Mutation rate must be between 0 and 1.")

        self.perc_selected = perc_selected
        self.mutation_rate = mutation_rate
        self.mutation_noise = mutation_noise
        self.elite = elite
        super(GeneticAlgorithm, self).__init__()

    def selection(self, scores, population_size):
        """
        Select the best performing agents in the previous generation as new parents

        :param scores:
        :type scores:
        :param population_size:
        :type population_size:
        :return:
        """
        n_selected = int(self.perc_selected * population_size)
        selected_parents = np.argsort(-scores)[:n_selected]
        return selected_parents

    def crossover(self, agents_weights, scores, population_size, sample_uniformly=False):
        """

        :param agents_weights:
        :param scores:
        :param population_size:
        :param sample_uniformly:
        :return:
        """

        # scale scores so to avoid negative values
        parents_scores = scores - (np.min(scores)) + 1

        if not sample_uniformly:
            # normalize scores and compute probabilities
            parents_scores_norm = parents_scores / (parents_scores.max() - parents_scores.min())
            parents_prob = parents_scores_norm / np.sum(parents_scores_norm)

        if self.elite:
            n_children = population_size - self.elite
        else:
            n_children = population_size
        children_weights = np.empty(n_children, dtype=np.ndarray)
        for i, c in enumerate(children_weights):
            prob = None if sample_uniformly else parents_prob
            parent1, parent2 = agents_weights[np.random.choice(len(agents_weights), size=2, replace=False, p=prob)]
            c = parent1
            for k, weights in enumerate(c):
                for index, w in np.ndenumerate(weights):
                    if np.random.uniform() > 0.5:
                        c[k][index] = parent2[k][index]
            children_weights[i] = c

        return children_weights

    def mutation(self, agent):
        """
        Mutate the previously generated children with random noise

        :param agent: one agent's weights
        :type agent: array of ndarray, as returned by Keras with get_weights()
        :return: mutated agent's weights
        """
        mutated = []
        for weights in agent:
            for i, w in np.ndenumerate(weights):
                if self.mutation_rate > np.random.uniform():
                    weights[i] = np.random.normal(w, self.mutation_noise)
            mutated.append(weights)
        return mutated

    def elitism(self, population, generation):
        """
        Select the best agents to keep for the next generation

        :param population:
        :type population:
        :param generation: generation index
        :type generation: int
        :return: best agents
        """
        scores = population.scores[generation]
        if np.count_nonzero(scores == max(scores)) > self.elite:
            best_index = np.where(scores == max(scores))[0]
            best_index = best_index[np.random.choice(len(best_index), size=self.elite, replace=False)]
        else:
            best_index = np.argsort(-(population.scores[generation]))[:self.elite]
        return population.agents_weights[generation][best_index]

    def generate_next_generation(self, population, generation):
        """
        Generate the next generation

        :param population:
        :type population:
        :param generation: generation index
        :type generation: int
        :return: new agents
        """

        id = self.selection(scores=population.scores[generation], population_size=population.size)
        new_children = self.crossover(agents_weights=population.agents_weights[generation][id],
                                      scores=population.scores[generation, id],
                                      population_size=population.size)

        next_generation = generation + 1
        for i, c in enumerate(new_children):
            population.agents_weights[next_generation][i] = self.mutation(c)
        if self.elite:
            best = self.elitism(population, generation)
            for j, b in enumerate(best):
                population.agents_weights[next_generation][i + j + 1] = b

