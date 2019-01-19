# import scipy
from scipy.spatial.distance import euclidean

from population import Agent
from visualization import *

plt.rcParams['figure.figsize'] = [12, 8]
font = {'family': 'serif',
        'weight': 'normal',
        'size': 16}
matplotlib.rc('font', **font)


def perturbate_weights(weights, n=5, noise_scale=1, render=False):
    """
    Randomly perturbate final weights, to verify the optimality condition

    :param weights: weights of the agent
    :type weights: array of weights and bias of neural network's layers
    :param n: number of perturbations
    :type n: int
    :param noise_scale: gaussian noise scale
    :type noise_scale: int or float
    :param render: animate the agent
    :type render: bool
    """
    for i in range(n):
        noise = np.random.normal(scale=noise_scale)
        agent = Agent(weights=weights + noise)
        score = agent.run_agent(render=render)
        print("{} - score: {}".format(i, score))


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
    weighted_initial_weights = np.dot((1 - alphas), initial_weights)

    # now each row has (alpha_i)w1, (alpha_i)w2 ...
    weighted_final_weights = np.dot(alphas, final_weights)

    new_weights = weighted_initial_weights + weighted_final_weights
    # we need a random agent
    agent = Agent()
    objective_values = np.apply_along_axis(lambda row: objective(row, agent), axis=1, arr=new_weights)

    return objective_values, alphas


def from_agent_to_weights(agent):
    """
    given and agent, we return a list of all its weights "flattened".
    :param agent: the agent for which we want to extract the weights
    :return: an numpy array containing the weights
    """
    weights = agent.model.get_weights()
    flattened_weights = []
    for layer in weights:
        if len(layer.shape) == 2:
            layer = layer.reshape(layer.shape[0] * layer.shape[1], )
        flattened_weights.extend(layer.tolist())
    return np.array(flattened_weights)


def from_weights_to_layers(weights, agent):
    """
    given the weights and an agent, it reconstructs the layers composing the agent's network architecture
    :param weights: the weights to use to build the layers
    :param agent: the agent whose architecture is of interest
    :return: the new layers, in a way that can be used by keras_model.set_weights(new_layers)
    """
    agent_weights = agent.model.get_weights()
    initial_point = 0
    new_layers = []
    for i, layer in enumerate(agent_weights):
        if len(layer.shape) == 2:
            new_layer = weights[initial_point:(initial_point + (layer.shape[0] * layer.shape[1]))]
            new_layer = new_layer.reshape(layer.shape[0], layer.shape[1])
            initial_point += layer.shape[0] * layer.shape[1]
        else:
            new_layer = weights[initial_point:(initial_point + layer.shape[0])]
            initial_point += layer.shape[0]
        new_layers.append(new_layer)
    return new_layers


def execute_agent_multiple_times(weights, agent, n_times=5):
    new_layers = from_weights_to_layers(weights, agent)
    agent.model.set_weights(new_layers)
    scores = [agent.run_agent() for _ in range(n_times)]
    return np.mean(scores)


def compute_distance_generations(weights):
    """
    We calculate the distances from the average of the weights (as a vector without considering bias) for each generation,
    distances are calculated between consecutive generations and for each generation from the first one (initialization).
    :param weights, np.array((n_generations, n_agents, n_weights)) returned by GA.run_agent_genetic
    :return:
        consecutive_dist: list, distances between generation
        dist_from_init: list, distances from initialization
    """
    w_1 = np.mean(weights[:, :, 0], 1).tolist()
    w_2 = np.mean(weights[:, :, 2], 1).tolist()
    flattened_mean_weights = []
    for i in range(len(w_1)):
        w_1[i].reshape(w_1[i].shape[0] * w_1[i].shape[1], )
        w_2[i].reshape(w_2[i].shape[0] * w_2[i].shape[1], )
        flattened_mean_weights.append(w_1[i].reshape(w_1[i].shape[0] * w_1[i].shape[1], ).tolist() + w_2[i].reshape(
            w_2[i].shape[0] * w_2[i].shape[1], ).tolist())
    consecutive_dist = []
    dist_from_init = []
    for first, second in zip(flattened_mean_weights, flattened_mean_weights[1:]):
        consecutive_dist.append(euclidean(first, second))
    for i, _ in enumerate(flattened_mean_weights):
        dist_from_init.append(euclidean(flattened_mean_weights[0], flattened_mean_weights[i]))

    return consecutive_dist, dist_from_init


def compute_distance_episodes(w0, w_t, w_t_new):
    """

    :param w0:
    :param w_t:
    :param w_t_new:
    :return:
    """

    distance_consecutive = euclidean(w_t.ravel(), w_t_new.ravel())
    distance_from_initial = euclidean(w0.ravel(), w_t_new.ravel())

    return distance_consecutive, distance_from_initial


def compute_second_derivative(i1, i2, loss_f, point, epsilon):
    """
    computes the second derivative of loss_f in point, with respect to the i1-th and i2-th directions
    :param i1: first direction
    :type i1: int
    :param i2: second direction
    :type i2: int
    :param loss_f: function for which we want to compute the second derivative
    :param point: point of evaluation
    :param epsilon: approximation constant (should be small)
    :return: a float number (the second derivative)
    """
    p1 = np.copy(point)
    p2 = np.copy(point)
    p3 = np.copy(point)
    p4 = np.copy(point)
    p1[i1] = point[i1] + epsilon
    p1[i2] = p1[i2] + epsilon
    p2[i1] = point[i1] - epsilon
    p2[i2] = p2[i2] + epsilon
    p3[i1] = point[i1] + epsilon
    p3[i2] = p3[i2] - epsilon
    p4[i1] = point[i1] - epsilon
    p4[i2] = p4[i2] - epsilon
    first_d1 = (loss_f(p1) - loss_f(p2)) / (2 * epsilon)
    first_d2 = (loss_f(p3) - loss_f(p4)) / (2 * epsilon)
    sec_deriv = (first_d1 - first_d2) / (2 * epsilon)
    return sec_deriv


def compute_hessian(loss_f, point, epsilon=0.01, file=None):
    """
    computes the hessian matrix of function loss_f in the point "point"
    :param loss_f: the function for which we want to compute the Hessian
    :param point: the point of evaluation of the hessian
    :param epsilon: approximation constant to compute the second derivative
    :param file: if string, after every row of the Hessian is computed, the matrix is saved to disk
    :return: the Hessian matrix
    """
    hessian = np.empty(shape=(point.shape[0], point.shape[0]))
    for i in range(point.shape[0]):
        for j in range(point.shape[0]):
            hessian[i][j] = compute_second_derivative(i, j, loss_f, point, epsilon)
        print("finished row {}/{}".format(i + 1, point.shape[0]))
        if file is not None:
            np.save("temp", hessian)
    if file is not None:
        np.save(file, hessian)
    return hessian


def get_top_eigenvector(hessian):
    """
    Finds the eigenvectors corresponding to the two larger eigenvalues
    :param hessian: the matrix for which we want to compute top eigenvectors
    :return: the eigenvectors corresponding to the two larger eigenvalues
    """
    eigvalues, eigvectors = np.linalg.eig(hessian)
    # order eigenvalues in descending order
    orderded_indices = np.argsort(eigvalues)[::-1]
    # get the eigenvectors corresponding to the top 2 eigenvalues
    max_eigvector = eigvectors[:, orderded_indices[0]]
    second_max_eig_vector = eigvectors[:, orderded_indices[1]]
    return max_eigvector, second_max_eig_vector


def plot_surface_3d(X, Y, Z):
    """
    plot 3d surface
    :param X: first axis
    :param Y: second axis
    :param Z: function evaluated at each of x,y points
    :return: the surface, and show the plot
    """
    X, Y = np.meshgrid(X, Y)
    Z = np.array(Z).reshape(X.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    plt.show()
    return surf


def evaluate_in_neighborhood(f, weights, d1, d2, n_steps=20):
    """
    Evaluate function f in a neighborhood of it around weights along directions d1 and d2
    :param f: the function of interest
    :param weights: the point of evaluation
    :param d1: the first direction of perturbation
    :param d2: the second direction of perturbation
    :param n_steps: the highest, the highest the precision
    :return: the function evaluated at all the points, and an array containing
            the magnitudes of perturbation
    """
    alphas = np.linspace(-2.0, 2.0, n_steps)
    scores = []
    print("evaluating the neighborhood")
    for i, a1 in enumerate(alphas.tolist()):
        for a2 in alphas.tolist():
            score = f(weights + d1 * a1 + d2 * a2)
            scores.append(score)
        print("finished step {}/{}".format(i + 1, alphas.shape[0]))
    return np.array(scores), alphas


def compute_epsilon_threshold(scores, epsilon=0.70, n_steps=80):
    """
    Given the scores in the neighborhood of the maximum reached by the training procedure, we want
    to evaluate its robustness. Therefore we compute the minimal perturbation along the directions
    of maximum curvature that make the agent perform arbitrarily bad (as indicated by epsilon).
    :param scores: the scores in the neighborhood of the agent (must be of size n_steps*n_steps)
    :param epsilon: the threshold of the score is computed as (max(scores)*epsilon).
    :param n_steps: the number of perturbation steps along each direction
    :return: the indexes of scores corresponding to the perturbation computed, the scores and threshold
    """
    assert scores.shape[0] == n_steps * n_steps
    assert 0 < epsilon <= 1

    max_index = int(n_steps / 2)
    scores = np.reshape(scores, (n_steps, n_steps))
    maximum = scores[max_index, max_index]
    threshold = maximum * epsilon
    s2 = scores[max_index, :]
    i = 0
    for i, _ in enumerate(s2.tolist()):
        s = s2[max_index + i]
        if s <= threshold:
            i = max_index + i
            break
        s = s2[max_index - i]
        if s <= threshold:
            i = max_index - i
            break

    s1 = scores[:, max_index]
    j = 0
    for j, _ in enumerate(s1.tolist()):
        s_ = s1[max_index + j]
        if s_ <= threshold:
            j = max_index + j
            break
        s_ = s1[max_index - j]
        if s_ <= threshold:
            j = max_index - j
            break
    return j, i, s1, s2, threshold


def compute_reward_along_eigenvectors(final_agent, n_times=2, file="hessian", from_file=False):
    """
    Plots the reward function along the directions of maximum variation of the curvature. These
    directions are defined by the eigenvectors corresponding to the two largest eigenvalues of the hessian.
    Use this function to evaluate the robustness of the agent.
    :param final_agent: the trained agent
    :param n_times: the reward function has of course no analytical form. Therefore to evaluate it
                    we run the agent and see the score. We want to take into account different possible
                    initial conditions, and we do it by running the agent multiple times and average the scores.
                    Set this parameter larger than one only if the mapping weights -> reward is not deterministic
                    or if you may have different initial conditions at each execution.
    :param file: a file name for the hessian matrix
    :param from_file: if True, it will read the matrix as specified by the file parameter
    :return:
    """

    def run_agent_multiple_times(final_weights):
        score = execute_agent_multiple_times(final_weights, final_agent, n_times=n_times)
        return score

    final_weights = from_agent_to_weights(final_agent)
    print("weights number", final_weights.shape)

    if not from_file:
        print("computing hessian")
        h = compute_hessian(run_agent_multiple_times, final_weights, file=file)
        print("hessian computed: ")
    else:
        h = np.load(file + ".npy")

    v1, v2 = get_top_eigenvector(h)
    scores, alphas = evaluate_in_neighborhood(run_agent_multiple_times, final_weights, v1, v2, n_steps=80)
    plot_surface_3d(alphas, alphas, scores)
    return v1, v2, scores, alphas


if __name__ == '__main__':
    """
    Example of main in which the whole pipeline is computed and the graphs plotted
    """
    # from GA import run_agent_es
    from ES import EvolutionStrategies
    from population import Population
    import matplotlib.pyplot as plt
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # agents_weights, scores, children = run_agent_es(n_agents=20, n_generations=50)

    es = EvolutionStrategies()

    agents = Population(optimizer=es)
    agents_weights, scores = agents.evolve(save=True)

    children_first = es.crossover(agents_weights[0], scores[0])
    children_last = es.crossover(agents_weights[-1], scores[-1])

    initial_agent = Agent(weights=children_first)
    final_agent = Agent(weights=children_last)
    final_agent_copy = Agent(weights=children_last)
    v1, v2, all_scores, alphas = compute_reward_along_eigenvectors(final_agent, file="Hessian_larger")
    t1, t2, s1, s2, threshold = compute_epsilon_threshold(all_scores)
    # print(v1,v2)

    print("threshold along v1 and v2")
    print(alphas[t1], alphas[t2])

    consecutive_dist, dist_init = compute_distance_generations(agents_weights)

    # plt.plot(alphas, s1, 'r', label='reward function along eigenvector 1')
    # plt.plot(alphas, np.ones(alphas.shape) * threshold, 'b', label='epsilon threshold')
    # plt.vlines(alphas[t1], 0, threshold, linestyles='dashed', label="perturbation magnitude")
    # plt.ylabel('threshold: {}'.format(threshold))
    # # plt.xlabel('eps')
    # plt.grid(True)
    # plt.legend()
    # plt.title('function along eigenvector relative to largest eigenvalue')
    # plt.show()

    plot_reward_along_eingenvector(alphas, t1, s1, threshold,
                                   title="Function along eigenvector relative to largest eigenvalue")

    # plt.plot(alphas, s2, 'r', label='reward function along eigenvector 2')
    # plt.plot(alphas, np.ones(alphas.shape) * threshold, 'b', label='epsilon threshold')
    # plt.vlines(alphas[t2], 0, threshold, linestyles='dashed', label="perturbation magnitude")
    # plt.ylabel('threshold: {}'.format(threshold))
    # # plt.xlabel('eps')
    # plt.grid(True)
    # plt.legend()
    # plt.title('function along eigenvector relative to second largest eigenvalue')
    # plt.show()

    plot_reward_along_eingenvector(alphas, t2, s2, threshold,
                                   title="Function along eigenvector relative to second largest eigenvalue")

    print("interpolating")
    inter_results, alphas = interpolate(initial_agent, final_agent_copy, execute_agent_multiple_times, n_steps=80)
    np.save("interpolation_results", inter_results)

    # plt.scatter(alphas, inter_results)
    # plt.ylabel('Mean score')
    # plt.xlabel('Alpha')
    # plt.grid(True)
    # plt.title('Linear interpolation initial and final agent')
    # plt.show()

    plot_interpolation(alphas, inter_results)

    # plt.plot([x for x in consecutive_dist], 'r', label='Dist consecutive gen')
    # plt.plot([x for x in dist_init], 'b', label='Dist from init')
    # plt.ylabel('Distance')
    # plt.xlabel('Generation')
    # plt.grid(True)
    # plt.legend()
    # plt.title('Distance of mean weights between generations')
    # plt.show()

    plot_distances([x for x in consecutive_dist], [x for x in dist_init],
                   title='Distance of mean weights between generations', x_label='Generation')
