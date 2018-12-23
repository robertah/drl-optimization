import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_weights_mean(weights, title="Weights Mean over Generations", xlabel="Generations", ylabel="Weights Mean"):
    means = []
    for w in weights:
        means.append(np.mean(w))
    plt.figure(figsize=(20, 10))
    plt.plot(means)
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)


def plot_weights_2d(weights, title="Weights Mean over Generations", xlabel="Generations", ylabel="Weights Mean"):
    pca = PCA(n_components=2)
    weights_2d = pca.fit_transform(weights)
    plt.figure(figsize=(20, 10))
    plt.plot(weights_2d)
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)


def plot_weights_difference(weights, title="Weights Diffs over Generations", xlabel="Generations", ylabel="Weights"):
    diffs = []
    for i, w in enumerate(weights):
        if i != 0:
            diffs.append(np.mean(w) - np.mean(weights[i-1]))
    plt.figure(figsize=(20, 10))
    plt.plot(diffs)
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)

