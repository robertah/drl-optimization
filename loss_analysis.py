import numpy as np
from genetic_2 import Agent
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
    final_weights = initial_weights.reshape(1, final_weights.shape[0])

    # now each row has (1-alpha_i)w1, (1-alpha_i)w2 ...
    weighted_initial_weights = np.dot((1-alphas), initial_weights)

    # now each row has (alpha_i)w1, (alpha_i)w2 ...
    weighted_final_weights = np.dot(alphas, final_weights)

    new_weights = weighted_initial_weights + weighted_final_weights

    # we need a random agent
    agent = Agent()
    objective_values = np.apply_along_axis(lambda row: objective(row, agent), axis=1, arr=new_weights)

    return objective_values


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


def execute_agent_multiple_times(weights, agent, n_times=50):
    new_layers = from_weights_to_layers(weights, agent)
    agent.model.set_weights(new_layers)
    print("one objective: ")
    start = time.time()
    scores = [agent.run_agent() for _ in range(n_times)]
    end = time.time()
    print(end - start)
    return np.mean(scores)


if __name__ == '__main__':
    from genetic_2 import run_agent_genetic
    import matplotlib.pyplot as plt

    initial_agent, final_agent = run_agent_genetic(n_generations=20)

    print("training_finished")

    results = interpolate(initial_agent, final_agent, execute_agent_multiple_times, n_steps=50)

    np.save("interpolation_results", results)

    plt.plot(results)
    plt.show()
