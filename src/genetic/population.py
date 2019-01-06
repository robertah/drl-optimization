import sys
import numpy as np
from .agent import Agent

sys.path.append("..")
from utils import save_results
from config import RANDOM_SEED

if RANDOM_SEED:
    np.random.seed(RANDOM_SEED)


class Population:

    def __init__(self, population_size, max_generations, n_selected, mutation_rate, mutation_noise, elite=None):
        """

        :param population_size:
        :param max_generations:
        :param n_selected:
        :param mutation_rate:
        :param mutation_noise:
        :param elite:
        """
        if not (0 <= mutation_rate <= 1):
            raise ValueError("Mutation rate must be between 0 and 1.")
        self.population_size = population_size
        self.max_generations = max_generations
        self.n_selected = int(n_selected) if n_selected > 1 else int(n_selected * population_size)
        self.mutation_rate = mutation_rate
        self.mutation_noise = mutation_noise
        self.elite = elite
        self.agents_weights = np.empty((self.max_generations, self.population_size, len(Agent().model.get_weights())),
                                       dtype=np.ndarray)
        self.scores = np.empty((self.max_generations, self.population_size), dtype=float)

    def create_population(self):
        """
        Initialize population

        :return:
        """
        return [Agent() for _ in range(self.population_size)]

    def selection(self, generation):
        """
        Select the best performing agents in the previous generation as new parents

        :param generation: generation index
        :type generation: int
        :return:
        """
        selected_parents = np.argsort(-(self.scores[generation]))[:self.n_selected]
        return self.agents_weights[generation][selected_parents]

    def crossover(self, parents):
        """
        Generate new agents from the previously chosen best agents

        :param parents: selected best agents returned by selection
        :type parents: list of genetic.Agent
        :return:
        """
        if self.elite:
            n_children = self.population_size - self.elite
        else:
            n_children = self.population_size
        children_weights = np.empty(n_children, dtype=np.ndarray)
        for i, c in enumerate(children_weights):
            parent1, parent2 = parents[np.random.choice(len(parents), size=2, replace=False)]
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

    def elitism(self, generation):
        """
        Select the best agents to keep for the next generation

        :param generation: generation index
        :type generation: int
        :return: best agents
        """
        scores = self.scores[generation]
        if np.count_nonzero(scores == max(scores)) > self.elite:
            best_index = np.where(scores == max(scores))[0]
            best_index = best_index[np.random.choice(len(best_index), size=self.elite, replace=False)]
        else:
            best_index = np.argsort(-(self.scores[generation]))[:self.elite]
        return self.agents_weights[generation][best_index]

    def generate_next_generation(self, generation):
        """
        Generate the next generation

        :param generation: generation index
        :type generation: int
        :return: new agents
        """
        parents = self.selection(generation)
        new_children = self.crossover(parents)
        new_generation = np.empty(self.population_size, dtype=np.ndarray)
        for i, c in enumerate(new_children):
            new_generation[i] = self.mutation(c)
        if self.elite:
            best = self.elitism(generation)
            for j, b in enumerate(best):
                new_generation[i + j + 1] = b
        return new_generation

    def terminate(self, score_threshold=2000, perc_threshold=0.95, n_consecutive=5):
        """
        Check if conditions to terminate genetic algorithm are satified:
        terminate genetic, if ``perc_threshold`` of total agents have scored the ``score_threshold``, in the last
        ``n_consecutive`` runs.

        :param scores: array of scores updated to current generation
        :type scores: np.ndarray of shape (n_generations, n_agents)
        :param score_threshold: maximum score
        :type score_threshold: int or float
        :param perc_threshold: percentage of agents that should reach the ``score_threshold``
        :type perc_threshold: float between 0 and 1
        :param n_consecutive: number of last consecutive generations
        :type n_consecutive: int
        :return:
        """
        assert 0 < perc_threshold <= 1

        if self.scores.shape[0] >= n_consecutive:
            last_runs = self.scores[-n_consecutive:]
            max_scores_count = [np.count_nonzero(s == score_threshold) for s in last_runs]
            if all(msc > self.population_size * perc_threshold for msc in max_scores_count):
                return True
            else:
                return False
        return False

    def evolve(self, save=True):
        """
        Evolve agents through genetic algorithm

        :param save: save agents weights and scores
        :type save: bool
        :return:
        """
        agents = self.create_population()

        for i in range(self.max_generations):
            print("\n generation", i)
            if not self.terminate():
                self.agents_weights[i] = np.array([a.model.get_weights() for a in agents], dtype=np.ndarray)

                for j, agent in enumerate(agents):  # TODO parallelize
                    score = agent.run_agent()
                    self.scores[i][j] = score

                print("mean {} - max {}".format(np.mean(self.scores[i]), np.max(self.scores[i])))

                new_agents = self.generate_next_generation(i)
                for i, a in enumerate(agents):
                    agents[i].model.set_weights(new_agents[i])

            else:
                print(i)
                self.agents_weights = self.agents_weights[:i]
                print(self.agents_weights.shape)
                self.scores = self.scores[:i]
                break

        if save:
            save_results(self.agents_weights, self.scores)

        return self.agents_weights, self.scores

    def simple_evolve(self, save=True):
        """
        Uber's implementaion of genetic algorithm without crossover
        """
        pass
