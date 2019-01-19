import sys
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np

from config import LOGGER, ENVIRONMENT
from utils import print_scores, save_results


class EvolutionaryOptimizers(ABC):
    """
    Base Optimizer class, for evolutionary algorithm
    """

    @abstractmethod
    def generate_next_generation(self, population, generation):
        raise NotImplementedError

    @staticmethod
    def terminate(population, generation, perc_threshold=0.95):
        """
        Check if conditions to terminate genetic algorithm are satified:
        terminate genetic, if ``perc_threshold`` of total agents have scored the ``score_threshold``, in the last
        ``n_consecutive`` runs.

        :param population:
        :type population:
        :param generation: generation index
        :type generation: int
        :return:
        """

        score_threshold = ENVIRONMENT.solving_avg_score
        n_consecutive = ENVIRONMENT.solving_n_trials

        if generation >= n_consecutive:
            last_runs = population.scores[generation - n_consecutive:generation]
            max_scores_count = [np.count_nonzero(s >= score_threshold) for s in last_runs]
            if all(msc > population.size * perc_threshold for msc in max_scores_count):
                return True
        return False

    def evolve(self, population, save=True):
        """
        Evolve agents

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
                try:
                    population.agents_weights[i] = np.array([a.model.get_weights() for a in agents], dtype=np.ndarray)

                    for j, agent in enumerate(agents):  # TODO parallelize
                        score = agent.run_agent()
                        population.scores[i][j] = score

                    print_scores(i + 1, population.scores[i])

                    if save and (i + 1) % 50 == 0:
                        save_results(population.agents_weights[:i], population.scores[:i], timestamp)

                    if i < population.max_generations - 1:
                        self.generate_next_generation(population=population, generation=i)

                        for k, a in enumerate(agents):
                            agents[k].model.set_weights(population.agents_weights[i + 1][k])

                except KeyboardInterrupt:
                    LOGGER.log(environment=ENVIRONMENT.name,
                               timestamp=timestamp,
                               algorithm=self.__class__.__name__,
                               parameters=vars(self),
                               generations=i,
                               score=np.max(population.scores[i-1]))
                    save_results(population.agents_weights[:i-1], population.scores[:i-1], timestamp)
                    sys.exit()

            else:
                population.agents_weights = population.agents_weights[:i]
                population.scores = population.scores[:i]
                break

        if save:
            LOGGER.log(environment=ENVIRONMENT.name,
                       timestamp=timestamp,
                       algorithm=self.__class__.__name__,
                       parameters=vars(self),
                       generations=i,
                       score=np.max(population.scores[i]))
            save_results(population.agents_weights, population.scores, timestamp)

        return population.agents_weights, population.scores
