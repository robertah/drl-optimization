import numpy as np
import scipy
import time


def interpolate(initial_agent, final_agent, objective, n_steps):
    """
    interpolation method to evaluate the path from initial weights to final weights
    that the network undergo during optimization (according to arXiv:1412.6544).
    We compute J(theta) for theta = (1-a)*wi + a*wf.
    :param initial_agent: the starting weights of the interpolation evaluation
            (for instance initial weights of the network)
    :param final_agent: the final weights of the interpolation evaluation
            (for instance final weights of the network
    :param objective: objective function J. In our case it must be a function that given
            an agent and a list of layers that defines the structure of the neural network,
            returns a score for the agent with the specified weights
    :param n_steps: number of interpolation steps
    :return: a numpy array containing J(theta) for each a (array of length n_steps)
    """
    alphas = np.linspace(0, 1, n_steps).reshape(n_steps, 1)

    initial_weights = from_agent_to_weights(initial_agent)
    initial_weights = initial_weights.reshape(1, initial_weights.shape[0])

    final_weights = from_agent_to_weights(final_agent)
    final_weights = final_weights.reshape(1, final_weights.shape[0])

    # now each row has (1-alpha_i)w1, (1-alpha_i)w2 ...
    weighted_initial_weights = np.dot((1-alphas), initial_weights)

    # now each row has (alpha_i)w1, (alpha_i)w2 ...
    weighted_final_weights = np.dot(alphas, final_weights)

    new_weights = weighted_initial_weights + weighted_final_weights
    # we need a random agent
    agent = Agent()
    objective_values = np.apply_along_axis(lambda row: objective(row, agent), axis=1, arr=new_weights)

    return objective_values, alphas 


def from_agent_to_weights(agent):
    weights = agent.model.get_weights()
    flattened_weights = []
    for layer in weights:
        if len(layer.shape) == 2:
            layer = layer.reshape(layer.shape[0]*layer.shape[1],)
        flattened_weights.extend(layer.tolist())
    return np.array(flattened_weights)



def from_weights_to_layers(weights, agent):
    agent_weights = agent.model.get_weights()
    initial_point = 0
    new_layers = []
    for i, layer in enumerate(agent_weights):
        if len(layer.shape) == 2:
            new_layer = weights[initial_point:(initial_point+(layer.shape[0]*layer.shape[1]))]
            new_layer = new_layer.reshape(layer.shape[0], layer.shape[1])
            initial_point += layer.shape[0]*layer.shape[1]
        else:
            new_layer = weights[initial_point:(initial_point + layer.shape[0])]
            initial_point += layer.shape[0]
        new_layers.append(new_layer)
    return new_layers


def execute_agent_multiple_times(weights, agent, n_times=20):
    new_layers = from_weights_to_layers(weights, agent)
    agent.model.set_weights(new_layers)
    print("one objective: ")
    start = time.time()
    scores = [agent.run_agent() for _ in range(n_times)]
    end = time.time()
    print(end - start)
    return np.mean(scores)


def distances_gen(weights):
    """
    We calculate the distances from the average of the weights (as a vector without considering bias) for each generation,
    distances are calculated between consecutive generations and for each generation from the first one (initialization).
    :param weights, np.array((n_generations, n_agents, n_weights)) returned by genetic.run_agent_genetic
    :return:
        consecutive_dist: list, distances between generation
        dist_from_init: list, distances from initialization
    """
    w_1 = np.mean(agents_weights[:,:,0],1).tolist()
    w_2 = np.mean(agents_weights[:,:,2],1).tolist()
    flattened_mean_weights = []
    for i in range(len(w_1)):
        w_1[i].reshape(w_1[i].shape[0] * w_1[i].shape[1], )
        w_2[i].reshape(w_2[i].shape[0] * w_2[i].shape[1], )
        flattened_mean_weights.append(w_1[i].reshape(w_1[i].shape[0] * w_1[i].shape[1], ).tolist() + w_2[i].reshape(w_2[i].shape[0] * w_2[i].shape[1], ).tolist() )
    consecutive_dist = []
    dist_from_init = []
    for first, second in zip(flattened_mean_weights, flattened_mean_weights[1:]):
        consecutive_dist.append(scipy.spatial.distance.euclidean(first, second))
    for i, _ in enumerate(flattened_mean_weights):
        dist_from_init.append(scipy.spatial.distance.euclidean(flattened_mean_weights[0], flattened_mean_weights[i]))

    return consecutive_dist, dist_from_init


if __name__ == '__main__':
    from genetic import run_agent_genetic
    from genetic.agent import Agent
    import matplotlib.pyplot as plt
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    agents_weights, scores, children = run_agent_genetic(n_agents=10, n_generations=3)

    initial_agent = Agent(weights=children[0])
    final_agent = Agent(weights=children[-1])
    consecutive_dist, dist_init = distances_gen(agents_weights)
    inter_results, alphas = interpolate(initial_agent, final_agent, execute_agent_multiple_times, n_steps=40)
    #np.save("run_results", run_results)
    np.save("interpolation_results", inter_results)

    plt.scatter(alphas, inter_results)
    plt.ylabel('Mean score')
    plt.xlabel('Alpha')
    plt.grid(True)
    plt.title('Linear interpolation initial and final agent')
    plt.show()

    plt.plot([x / max(consecutive_dist) for x in consecutive_dist], 'r', label='Dist consecutive gen')
    plt.plot([x / max(dist_init) for x in dist_init], 'b', label='Dist from init')
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.grid(True)
    plt.legend()
    plt.title('Distance of mean weights between generations ')
    plt.show()
