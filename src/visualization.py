from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA

from config import ENVIRONMENT, VISUALIZATION_WEIGHTS

plt.rcParams['figure.figsize'] = [12, 8]
font = {'family': 'serif',
        'weight': 'normal',
        'size': 16}
matplotlib.rc('font', **font)


def plot_weights_mean(weights, title="Weights Mean over Generations", xlabel="Generations", ylabel="Weights Mean"):
    """
    Plot the mean for each generation's weights

    :param weights: array of weights (ndarray) obtained during each generation
    :param title: plot's title
    :param xlabel: plot's x label
    :param ylabel: plot's y label
    """
    means = []
    for w in weights:
        means.append(np.mean(w))
    plt.plot(means)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_weights_2d(weights, scores, title='Weights Evolution', save=False):
    """
    Plot the weights reduced in 2d

    :param weights: array of weights (only one layer) of each agents
    :type weights: np.ndarray of shape (n_generations, n_agents, n_weights)
    :param scores: array of scores of each agent
    :type scores: np.ndarray of shape (n_generations, n_agents)
    :param title: plot's title
    :type title: str
    :param save: save the plot animation as gif
    :type save: bool
    """
    assert len(weights.shape) == 3

    n_generations, n_agents = weights.shape[0], weights.shape[1]

    pca = PCA(n_components=2)
    weights_2d = pca.fit_transform(weights.reshape(n_generations * n_agents, -1))
    weights_2d = weights_2d.reshape(n_generations, n_agents, 2)

    fig, ax = plt.subplots()
    xmin, xmax = np.min(weights_2d[:, :, 0]), np.max(weights_2d[:, :, 0])
    ymin, ymax = np.min(weights_2d[:, :, 1]), np.max(weights_2d[:, :, 1])
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

    norm = Normalize(0, scores.max())
    colormap = cm.ScalarMappable(norm, 'magma_r')
    colors = colormap.to_rgba(scores)

    scatter = ax.scatter(weights_2d[0, :, 0], weights_2d[0, :, 1], c=scores[0], cmap='magma_r', vmin=0,
                         vmax=scores.max())
    gen = ax.text(int(xmax - abs(xmax - xmin) / 2), int(ymax), "Generation: 0")
    cbar = plt.colorbar(scatter)
    cbar.set_label('Scores')
    plt.title(title)

    def update(i):
        gen.set_text("Generation: {}".format(i))
        scatter.set_offsets(np.c_[weights_2d[i, :, 0], weights_2d[i, :, 1]])
        scatter.set_color(colors[i])
        return scatter,

    anim = FuncAnimation(fig, update, frames=n_generations, interval=100)
    if save:
        anim.save(
            VISUALIZATION_WEIGHTS + '/{}-{}.gif'.format(ENVIRONMENT.name, datetime.now().strftime('%Y%m%d%H%M%S')),
            fps=30, writer='imagemagick')
    return anim


def plot_weights_difference(weights, title="Weights Diffs over Generations", xlabel="Generations", ylabel="Weights"):
    """
    Plot the difference of centroids of weights wrt to the previous generation's weights

    :param weights: array of weights (ndarray) obtained during each generation
    :param title: plot's title
    :param xlabel: plot's x label
    :param ylabel: plot's y label
    """
    diffs = []
    for i, w in enumerate(weights):
        if i != 0:
            diffs.append(np.mean(w) - np.mean(weights[i - 1]))
    plt.plot(diffs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_scores_generations(scores, title="Scores over generations", xlabel="Generations", ylabel="Scores"):
    """
    Plot the mean, std bands and max of scores obtained during training

    :param scores: scores (ndarray) obtained by each agent during each generation
    :type scores: np.ndarray of shape (n_generation, n_agents)
    :param title: plot's title
    :type title: str
    :param xlabel: plot's x label
    :type xlabel: str
    :param ylabel: plot's y label
    :type ylabel: str
    """
    x_ticks = np.arange(len(scores))
    means = np.mean(scores, axis=1)
    stds = np.std(scores, axis=1)
    maxs = np.max(scores, axis=1)
    plt.plot(means, label="mean")
    plt.fill_between(x_ticks, means - stds, means + stds, color="grey", alpha=0.2, label="standard deviation")
    plt.plot(maxs, label="max")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()


def plot_scores_episodes(scores, title="Scores over episodes", file="scores.pdf"):
    """
    Plot the mean, std bands and max of scores obtained during training

    :param scores: scores (ndarray) obtained by each agent during each generation
    :type scores: np.ndarray of shape (n_generation, n_agents)
    :param title: plot's title
    :type title: str
    :param xlabel: plot's x label
    :type xlabel: str
    :param ylabel: plot's y label
    :type ylabel: str
    :param file: saving file
    """

    plt.title(title)
    plt.plot([i * 20 for i in range(0, len(scores))], scores)
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.savefig(file)


def plot_distances(between_consecutive, from_initial, x_label="Generation", title='Distances', file='distance.pdf'):
    """
    Plot weights distances between consecutive episodes/generation and from the initail weights.

    :param between_consecutive: array of computed distances between consecutive weights
    :param from_initial: array of computed distances from initial weights
    :param x_label: plot x label
    :param title: plot's title
    :param file: saving file
    """

    plt.title(title)
    plt.plot(between_consecutive, 'r', label='Between consecutive')
    plt.plot(from_initial, 'b', label='From initial')
    plt.legend()
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel('Euclidean distance')
    plt.show()
    plt.savefig(file)


def plot_interpolation(alphas, inter_results, title='Linear interpolation initial and final agent',
                        file='interpolation.pdf'):
    """

    :param alphas: computed alphas
    :param inter_results: computed interpolation results
    :param title: plot's title
    :param file: saving file
    :return:
    """
    plt.scatter(alphas, inter_results)
    plt.xlabel('Alpha')
    plt.ylabel('Mean score')
    plt.grid(True)
    plt.title(title)
    plt.show()
    plt.savefig(file)


def plot_reward_along_eingenvector(alphas, t, scores, threshold,
                                   title='Function along eigenvector relative to largest eigenvalue',
                                   file='reward_engeinvector.pdf'):
    """

    :param alphas:
    :param t:
    :param scores:
    :param threshold:
    :param title:
    :param file:
    :return:
    """

    plt.plot(alphas, scores, 'r', label='Reward function along eigenvector')
    plt.plot(alphas, np.ones(alphas.shape)*threshold, 'b', label='Epsilon threshold')
    plt.vlines(alphas[t], 0, threshold, linestyles='dashed', label="Perturbation magnitude")
    plt.ylabel('threshold: {}'.format(threshold))
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.show()
    plt.savefig(file)
