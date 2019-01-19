from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
import matplotlib

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
    plt.figure(figsize=figsize)
    plt.plot(means)
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)


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

    fig, ax = plt.subplots(figsize=figsize)
    xmin, xmax = np.min(weights_2d[:, :, 0]), np.max(weights_2d[:, :, 0])
    ymin, ymax = np.min(weights_2d[:, :, 1]), np.max(weights_2d[:, :, 1])
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

    norm = Normalize(0, scores.max())
    colormap = cm.ScalarMappable(norm, 'magma_r')
    colors = colormap.to_rgba(scores)

    scatter = ax.scatter(weights_2d[0, :, 0], weights_2d[0, :, 1], c=scores[0], cmap='magma_r', vmin=0, vmax=scores.max())
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
    plt.figure(figsize=figsize)
    plt.plot(diffs)
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)


def plot_scores(scores, title="Scores over generations", xlabel="Generations", ylabel="Scores"):
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
    plt.figure(figsize=figsize)
    plt.plot(means, label="mean")
    plt.fill_between(x_ticks, means - stds, means + stds, color="grey", alpha=0.2, label="standard deviation")
    plt.plot(maxs, label="max")
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.legend(fontsize=10)
