from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime
from utils import print_scores, save_results


class EvolutionaryOptimizers(ABC):

    @abstractmethod
    def generate_next_generation(self):
        raise NotImplementedError

    @staticmethod
    def terminate(population, generation, score_threshold=2000, perc_threshold=0.95, n_consecutive=5):
        """
        Check if conditions to terminate genetic algorithm are satified:
        terminate genetic, if ``perc_threshold`` of total agents have scored the ``score_threshold``, in the last
        ``n_consecutive`` runs.

        :param population:
        :type population:
        :param generation: generation index
        :type generation: int
        :param score_threshold: maximum score
        :type score_threshold: int or float
        :param perc_threshold: percentage of agents that should reach the ``score_threshold``
        :type perc_threshold: float between 0 and 1
        :param n_consecutive: number of last consecutive generations
        :type n_consecutive: int
        :return:
        """
        assert 0 < perc_threshold <= 1

        if generation >= n_consecutive:
            last_runs = population.scores[generation - n_consecutive:generation]
            max_scores_count = [np.count_nonzero(s == score_threshold) for s in last_runs]
            if all(msc > population.size * perc_threshold for msc in max_scores_count):
                return True
        return False

    def evolve(self, population, save=True):
        """
        Evolve agents through genetic algorithm

        :param population:
        :type population:
        :param save: save agents weights and scores
        :type save: bool
        :return:
        """

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        print("Optimization - started", timestamp)

        agents = population.create_population()

        for i in range(population.max_generations):
            if not self.terminate(population, i):
                population.agents_weights[i] = np.array([a.model.get_weights() for a in agents], dtype=np.ndarray)

                for j, agent in enumerate(agents):  # TODO parallelize
                    score = agent.run_agent()
                    population.scores[i][j] = score

                print_scores(i + 1, population.scores[i])

                self.generate_next_generation(population=population, generation=i)
                for k, a in enumerate(agents):
                    agents[k].model.set_weights(population.agents_weights[i + 1][k])

                if save and (i + 1) % 100 == 0:
                    save_results(population.agents_weights, population.scores, timestamp)

            else:
                print(i)
                population.agents_weights = population.agents_weights[:i]
                print(population.agents_weights.shape)
                population.scores = population.scores[:i]
                break

        if save:
            save_results(population.agents_weights, population.scores, timestamp)

        return population.agents_weights, population.scores
