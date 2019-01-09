# -*- coding: utf-8 -*-
from .agent import Agent, Environment
import numpy as np
import threading
from .genetic_functions import crossover_function, crossover_function_1, generate_population, crossover_function_2, generate_population_2, crossover_function_1_1, noise, generate_population_1, new_population, new_population_elite
from utils import save_results

def run_agent_genetic( n_agents=50, n_generations=100, save=True):
    n_weights = len(Agent().model.get_weights())

    agents_weights = np.empty((n_generations, n_agents, n_weights), dtype=np.ndarray)
    scores = np.empty((n_generations, n_agents), dtype=float)

    children = np.empty((n_generations, n_weights), dtype=np.ndarray)

    # initialize agents
    agents = [Agent() for _ in range(n_agents)]

    for i in range(n_generations):
        agents_weights[i] = np.array([a.model.get_weights() for a in agents], dtype=np.ndarray)
        print(i)

        for j, agent in enumerate(agents):  # TODO parallelize
            scores[i][j] = agent.run_agent()

        child = crossover_function(agents, scores[i])
        children[i] = np.array(child, dtype=np.ndarray).reshape(n_weights)
        agents = generate_population(child, n_agents, agents)

    if save:
        save_results(agents_weights, scores)

    return agents_weights, scores, children

def run_agent_genetic_alternative(n_init_agents=300, n_agents=50, n_generations=100, save=True):
    '''
    In this version we have a different number of agents in the inizialization in order to better expore the parameter space at the beginning.
    This is usefull just in the inizialization "to take the right direction in the parameter space" so it would be useless and computationally expensive use so much agents once the alg is initialized.
    :param n_init_agents:
    :param n_agents:
    :param n_generations:
    :param save:
    :return:
    '''
    n_weights = len(Agent().model.get_weights())

    agents_weights = np.empty((n_generations-1, n_agents, n_weights), dtype=np.ndarray)
    agents_init_weights = np.empty((n_init_agents, n_weights), dtype=np.ndarray)
    scores = np.empty((n_generations-1, n_agents), dtype=float)
    init_scores = np.empty((n_init_agents), dtype=float)

    children = np.empty((n_generations, n_weights), dtype=np.ndarray)

    # initialize agents
    init_agents = [Agent() for _ in range(n_init_agents)]
    agents = [Agent() for _ in range(n_agents)]

    for i in range(n_generations):
        print(i)
        if i == 0:
            agents_init_weights = np.array([a.model.get_weights() for a in init_agents], dtype=np.ndarray)
            for j, agent in enumerate(init_agents):  # TODO parallelize
                init_scores[j] = agent.run_agent()
            child = crossover_function(init_agents, init_scores)

        else:
            agents_weights[i-1] = np.array([a.model.get_weights() for a in agents], dtype=np.ndarray)
            for j, agent in enumerate(agents):  # TODO parallelize
                scores[i-1][j] = agent.run_agent()
            child = crossover_function(agents, scores[i-1])


        children[i] = np.array(child, dtype=np.ndarray).reshape(n_weights)
        agents = generate_population(child, n_agents, agents)

    if save:
        save_results(agents_weights, scores)

    return agents_weights, scores, children

def run_agent_genetic_2( n_agents=50, n_generations=50, save=True):
    #This use crossover function 2 and generate popuation 2 instead of 1
    n_weights = len(Agent().model.get_weights())

    agents_weights = np.empty((n_generations, n_agents, n_weights), dtype=np.ndarray)
    scores = np.empty((n_generations, n_agents), dtype=float)

    #children = np.empty((n_generations, n_weights), dtype=np.ndarray)

    # initialize agents
    agents = [Agent() for _ in range(n_agents)]

    for i in range(n_generations):
        agents_weights[i] = np.array([a.model.get_weights() for a in agents], dtype=np.ndarray)
        print(i)

        for j, agent in enumerate(agents):  # TODO parallelize
            scores[i][j] = agent.run_agent()

        child1,child2 = crossover_function_2(agents, scores[i])
        #children[i] = np.array(child, dtype=np.ndarray).reshape(n_weights)
        agents = generate_population_2(child1, child2, n_agents, agents)

    if save:
        save_results(agents_weights, scores)

    return agents_weights, scores

def run_agent_genetic_positive( n_agents=50, n_generations=100, save=True):
    '''
    This function uses crossover_function_1_1 where a linear interpolation between 0 and 1 is performed to have all scores (positive and negative)
    scaled and positive between 0 and 1. To give more importance to better scores, the 3 power of the scores is taken before normalization.
    '''
    # initialize environment
    env = Environment()

    n_weights = len(Agent(env).model.get_weights())

    agents_weights = np.empty((n_generations, n_agents, n_weights), dtype=np.ndarray)
    scores = np.empty((n_generations, n_agents), dtype=float)
    noises = np.empty(n_generations, dtype=float)
    children = np.empty((n_generations, n_weights), dtype=np.ndarray)

    # initialize agents
    agents = [Agent(env) for _ in range(n_agents)]

    for i in range(n_generations):
        agents_weights[i] = np.array([a.model.get_weights() for a in agents], dtype=np.ndarray)
        print(i)

        for j, agent in enumerate(agents):  # TODO parallelize
            scores[i][j] = agent.run_agent(env)

        child = crossover_function_1_1(agents, scores[i])
        #children[i] = np.array(child, dtype=np.ndarray).reshape(n_weights)
        agents = generate_population_1(child, n_agents, agents)
        noises[i] = noise(scores[i])
        '''
        child1,child2 = crossover_function_2(agents, scores[i])
        agents = generate_population_2(child1, child2, n_agents, agents, noise(scores[i]))
        noises[i] = noise(scores[i])
        '''
        print(scores[i])
        print('max_rew:', scores[i].max(), 'avarage_rew:', np.average(scores[i]))
    if save:
        save_results(scores)

    return agents_weights, scores, children

def run_genetic_no_mean( n_agents=50, n_generations=100, save=True):
    # initialize environment
    env = Environment()

    n_weights = len(Agent(env).model.get_weights())

    #agents_weights = np.empty((n_generations, n_agents, n_weights), dtype=np.ndarray)
    scores = np.empty((n_generations, n_agents), dtype=float)
    #children = np.empty((n_generations, n_weights), dtype=np.ndarray)
    l = [40,80,100,120,140,160,180,200,300,400]
    # initialize agents
    agents = [Agent(env) for _ in range(n_agents)]

    for i in range(n_generations):
        #agents_weights[i] = np.array([a.model.get_weights() for a in agents], dtype=np.ndarray)
        print(i)

        for j, agent in enumerate(agents):  # TODO parallelize
            if i in l:
                scores[i][j] = agent.run_agent(env, render = False)
            else:
                scores[i][j] = agent.run_agent(env)

        agents = new_population(agents,scores[i],4)
        print('max_rew:', scores[i].max(), 'avarage_rew:', np.average(scores[i]))
        if i in l:
            save_results(scores)
    if save:
        save_results(scores)

    return scores

#MULTI THREADS:
#works for all sub agents list of the same lenght
def run_some_agents(agents, thread_index, scores,i,env):
    for j, agent in enumerate(agents):
        scores[i][j+ thread_index*len(agents)] = agent.run_agent(env)

def run_genetic_no_mean_parallel( n_agents=100, n_generations=300, save=True):
    # initialize environment
    env0 = Environment()
    env1 = Environment()
    env2 = Environment()
    env3 = Environment()
    n_weights = len(Agent(env1).model.get_weights())
    n_threads = 4
    #agents_weights = np.empty((n_generations, n_agents, n_weights), dtype=np.ndarray)
    scores = np.empty((n_generations, n_agents), dtype=float)
    noises = np.empty(n_generations, dtype=float)
    #children = np.empty((n_generations, n_weights), dtype=np.ndarray)
    l = [20,40,60,80,100,120]
    # initialize agents
    agents = [Agent(env1) for _ in range(n_agents)]

    for i in range(n_generations):
        #agents_weights[i] = np.array([a.model.get_weights() for a in agents], dtype=np.ndarray)
        print(i)

        thread0 = threading.Thread(target=run_some_agents, args=(agents[0: n_agents // n_threads - 1], 0,scores,i,env0))
        thread1 = threading.Thread(target=run_some_agents, args=(agents[n_agents // n_threads : n_agents * 2 // n_threads -1], 1, scores, i, env1))
        thread2 = threading.Thread(target=run_some_agents, args=(agents[n_agents*2 // n_threads : n_agents * 3 // n_threads -1], 2, scores, i, env2))
        thread3 = threading.Thread(target=run_some_agents, args=(agents[n_agents * 3 // n_threads :n_agents - 1], 3, scores, i, env3))

        thread0.start()
        thread1.start()
        thread2.start()
        thread3.start()

        agents = new_population(agents,scores[i],5)

        print('max_rew:', scores[i].max(), 'avarage_rew:', np.average(scores[i]))

        if i in l:
            save_results(scores)
    if save:
        save_results(scores)

    return scores