# -*- coding: utf-8 -*-
import numpy as np
import sys
from .agent import Agent
from math import exp
import random


from config import RANDOM_SEED, GA_MUTATION_NOISE

if RANDOM_SEED:
    np.random.seed(RANDOM_SEED)

def noise(rewards):
    '''
    This is a test to have a scalable sigmoidal noise to have better exporation depending on the maximum achived score.
    Not sure if it is better to use the max or the average reward.
    The noise function is create ad hoc for the problem.
    :param rewards:
    :return: noise
    '''
    noise = 80 /(1+exp(0.03*(rewards.max()))) + 0.1
    print('max_rew:', rewards.max(), 'avarage_rew:', np.average(rewards) ,'  noise:',  noise)
    return noise


def crossover_function(agents, rewards):
    '''
    agents: a list of keras neural networks (parents agents)
    rewards: list (or array) of rewards associated to the performance of the
          corresponding model
    return: the child weights computed by the weighted average
          of the parents w.r.t. the reward
    '''
    rewards = np.array(rewards)
    num_layers = len(agents[0].model.get_weights())
    normalized_rewards = rewards / np.sum(rewards)
    child_model = []
    for i in range(num_layers):
        new_layer = np.zeros_like(agents[0].model.get_weights()[i])
        for j, parent_agent in enumerate(agents):
            layer = parent_agent.model.get_weights()[i] * normalized_rewards[j]
            new_layer = new_layer + layer
        child_model.append(new_layer)
    return child_model


def crossover_function_1(agents, reward):
    '''
    This one use only the model with a positive score to generate the next population
    models: a list of keras neural networks (parents agents)
    rewars: list (or array) of rewards associated to the performance of the
            corresponding model
    return: the child weights computed by the weighted average
            of the parents w.r.t. the reward
    '''
    index = 0
    noise = 0.1
    rewards = reward.tolist()
    agents_copy = agents.copy()
    while index < len(rewards):
        if rewards[index] < 0:
            del agents_copy[index]
            del rewards[index]
        else:
            index += 1

    rewards = np.array(rewards)
    if len(agents_copy) == 0:
        noise = 0.5
        #pick the meno peggio, return this as child model and increase the noise in the next gen
        return agents[np.argmax(reward)].model.get_weights()
        #sys.exit('Init Fail')

    num_layers = len(agents[0].model.get_weights())
    normalized_rewards = rewards / np.sum(rewards)
    child_model = []
    for i in range(num_layers):
        new_layer = np.zeros_like(agents_copy[0].model.get_weights()[i])
        for j, parent_agent in enumerate(agents_copy):
            layer = parent_agent.model.get_weights()[i] * normalized_rewards[j]
            new_layer = new_layer + layer
        child_model.append(new_layer)
    return child_model

def crossover_function_1_1(agents, rewards):
    '''
    agents: a list of keras neural networks (parents agents)
    rewards: list (or array) of rewards associated to the performance of the
          corresponding model



    return: the child weights computed by the weighted average
          of the parents w.r.t. the reward
    '''
    rewards = np.array(rewards)
    scaled_rewards = np.interp(rewards, (rewards.min(), rewards.max()), (0, 1))
    #scaled_rewards = np.exp(rewards)
    normalized_rewards = scaled_rewards/np.sum(scaled_rewards)
    num_layers = len(agents[0].model.get_weights())
    child_model = []
    for i in range(num_layers):
        new_layer = np.zeros_like(agents[0].model.get_weights()[i])
        for j, parent_agent in enumerate(agents):
            layer = parent_agent.model.get_weights()[i] * normalized_rewards[j]
            new_layer = new_layer + layer
        child_model.append(new_layer)
    return child_model

def crossover_function_1_1_normalized(agents, rewards):
    '''
    agents: a list of keras neural networks (parents agents)
    rewards: list (or array) of rewards associated to the performance of the
          corresponding model



    return: the child weights computed by the weighted average
          of the parents w.r.t. the reward
    '''
    rewards = np.array(rewards)
    scaled_rewards = np.power(np.interp(rewards, (rewards.min(), rewards.max()), (0, 1)),3)
    #scaled_rewards = np.exp(rewards)
    normalized_rewards = scaled_rewards/np.sum(scaled_rewards)
    num_layers = len(agents[0].model.get_weights())
    child_model = []
    for i in range(num_layers):
        new_layer = np.zeros_like(agents[0].model.get_weights()[i])
        for j, parent_agent in enumerate(agents):
            layer = parent_agent.model.get_weights()[i] * normalized_rewards[j]
            new_layer = new_layer + layer
        if new_layer.ndim == 1:
            child_model.append(new_layer / new_layer.sum(axis=0, keepdims=1))
        else:
            child_model.append(new_layer/new_layer.sum(axis=1, keepdims=1))
    return child_model

def crossover_function_1_2(agents, rewards):
    rewards = np.array(rewards)
    scaled_rewards = np.interp(rewards, (rewards.min(), rewards.max()), (0, 1))
    ordered_indexes = np.argsort(-scaled_rewards)
    best_agents = [agents[i] for i in ordered_indexes[:5] ]
    best_scaled_norm_rew = scaled_rewards[ordered_indexes[:5]]/np.sum(scaled_rewards[ordered_indexes[:5]])
    num_layers = len(agents[0].model.get_weights())
    child_model = []
    for i in range(num_layers):
        new_layer = np.zeros_like(agents[0].model.get_weights()[i])
        for j, parent_agent in enumerate(best_agents):
            layer = parent_agent.model.get_weights()[i] * best_scaled_norm_rew[j]
            new_layer = new_layer + layer
        child_model.append(new_layer/new_layer.sum(axis=1, keepdims=1))
    return  child_model

def new_population(agents,rewards,best):
    rewards = np.array(rewards)
    scaled_rewards = np.interp(rewards, (rewards.min(), rewards.max()), (0, 1))
    ordered_indexes = np.argsort(-scaled_rewards)
    best_agents = [agents[i] for i in ordered_indexes[:best] ]
    best_scaled_norm_rew = scaled_rewards[ordered_indexes[:best]]/np.sum(scaled_rewards[ordered_indexes[:best]])
    num_layers = len(agents[0].model.get_weights())
    for agent in agents:
        child_model = []
        parents = random.sample(range(best),2)
        for i in range(num_layers):
            layer_shape = agent.model.get_weights()[i].shape
            mother_father = np.random.choice([0,1],layer_shape)
            new_layer = np.multiply(best_agents[parents[0]].model.get_weights()[i],mother_father) + np.multiply(best_agents[parents[1]].model.get_weights()[i],np.ones(layer_shape)-mother_father)
            noise = np.multiply(new_layer, np.random.choice([-1, 0, 1], new_layer.shape, p=[0.3,0.4,0.3])) * random.uniform(0, 2)
            new_layer = new_layer + noise
            child_model.append(new_layer)
        agent.model.set_weights(child_model)
    return agents

def new_population_elite(agents,rewards,best):
    rewards = np.array(rewards)
    scaled_rewards = np.interp(rewards, (rewards.min(), rewards.max()), (0, 1))
    ordered_indexes = np.argsort(-scaled_rewards)
    best_agents = [agents[i] for i in ordered_indexes[:best]]
    new_agents = [agents[i] for i in ordered_indexes[best:]]
    best_scaled_norm_rew = scaled_rewards[ordered_indexes[:best]]/np.sum(scaled_rewards[ordered_indexes[:best]])
    num_layers = len(agents[0].model.get_weights())
    for agent in new_agents:
        child_model = []
        parents = random.sample(range(best),2)
        for i in range(num_layers):
            layer_shape = agent.model.get_weights()[i].shape
            mother_father = np.random.choice([0,1],layer_shape)
            new_layer = np.multiply(best_agents[parents[0]].model.get_weights()[i],mother_father) + np.multiply(best_agents[parents[1]].model.get_weights()[i],np.ones(layer_shape)-mother_father)
            noise = np.multiply(new_layer, np.random.choice([-1, 0, 1], new_layer.shape)) * random.uniform(0, 1)
            new_layer = new_layer + noise
            child_model.append(new_layer)
        agent.model.set_weights(child_model)
    new_agents = new_agents + best_agents
    return new_agents



def generate_population(child_model, num_children, agents, noise):
    """
    child_model: model from which building the new population
    num_children: number of children to generate
    scale_noise: variance of the gaussian noise to apply
    agents: list of agents
    """
    for child in range(num_children):
        new_child = []
        for layer in child_model:
            #new_layer = np.random.normal(layer, GA_MUTATION_NOISE)
            new_layer = np.random.normal(layer, noise)
            new_child.append(new_layer)
        # Ho fatto questa piccola modifica per avere direttamente una lista di agenti che  quello che poi ci servirebbe piuttosto che una lista di modelli ma non sono sicuro che funzioni
        agents[child].model.set_weights(new_child)
    return agents

def generate_population_1(child_model, num_children, agents):
    """
    child_model: model from which building the new population
    num_children: number of children to generate
    scale_noise: variance of the gaussian noise to apply
    agents: list of agents
    """
    for child in range(num_children):
        new_child = []
        for layer in child_model:
            noise = np.multiply(layer,np.random.choice([-1,0,1],layer.shape)) * random.uniform(0,10)
            new_layer = layer + noise
            new_child.append(new_layer)
        # Ho fatto questa piccola modifica per avere direttamente una lista di agenti che  quello che poi ci servirebbe piuttosto che una lista di modelli ma non sono sicuro che funzioni
        agents[child].model.set_weights(new_child)
    return agents


def crossover_function_2(agents, rewards):
    '''
    models: a list of keras neural networks (parents agents)
    rewars: list (or array) of rewards associated to the performance of the
          corresponding model
    return: the best two agents
    '''
    #sorted_indeces = np.argsort(-np.array(rewards))
    reward = np.array(rewards)
    best_agent = agents[np.argmax(reward)]
    reward[np.argmax(reward)] = -700
    second_best = agents[np.argmax(reward)]
    return best_agent, second_best


def generate_population_2(child_model1, child_model2, num_children, agents, noise):
    '''
    child_model: model from which building the new population
    num_children: number of children to generate
    scale_noise: variance of the gaussian noise to apply
    '''

    for child in range(int(num_children / 2) - 1):
        new_child = []
        for layer in child_model1.model.get_weights():
            new_layer = np.random.normal(layer, noise)
            new_child.append(new_layer)
        agents[child].model.set_weights(new_child)

    for child in range(int(num_children / 2) - 1, num_children - 2):
        new_child = []
        for layer in child_model2.model.get_weights():
            new_layer = np.random.normal(layer, noise)
            new_child.append(new_layer)
        agents[child].model.set_weights(new_child)

    agents[-2].model.set_weights(child_model1.model.get_weights())
    agents[-1].model.set_weights(child_model2.model.get_weights())
    return agents

