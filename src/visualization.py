import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


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
    plt.figure(figsize=(20, 10))
    plt.plot(means)
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)


def plot_weights_2d(weights, title="Weights Mean over Generations", xlabel="Generations", ylabel="Weights Mean"):
    """
    Plot the weights reduced in 2d

    :param weights: array of weights (ndarray) obtained during each generation
    :param title: plot's title
    :param xlabel: plot's x label
    :param ylabel: plot's y label
    """
    pca = PCA(n_components=2)
    weights_2d = pca.fit_transform(weights)  # TODO fix weights type
    plt.figure(figsize=(20, 10))
    plt.plot(weights_2d)
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)


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
    plt.figure(figsize=(20, 10))
    plt.plot(diffs)
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)


def plot_scores(scores, title="Scores over generations", xlabel="Generations", ylabel="Scores"):
    """
    Plot the mean, std bands and max of scores obtained during training

    :param scores: scores (ndarray) obtained by each agent during each generation (e.g. if using results returned by
                   run_agent_genetic(), scores = np.array(results[:, 1, :], dtype=float))
    :param title: plot's title
    :param xlabel: plot's x label
    :param ylabel: plot's y label
    """
    x_ticks = np.arange(len(scores))
    means = np.mean(scores, axis=1)
    stds = np.std(scores, axis=1)
    maxs = np.max(scores, axis=1)
    plt.figure(figsize=(20, 10))
    plt.plot(means, label="mean")
    plt.fill_between(x_ticks, means - stds, means + stds, color="grey", alpha=0.2, label="standard deviation")
    plt.plot(maxs, label="max")
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.legend(fontsize=16)
