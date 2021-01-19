from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import math


def entropy(Y):
    unique, counts = np.unique(Y, return_counts=True)
    entropy = 0
    for i, value in enumerate(unique):
        if unique[i] != value:
            continue
        prob = counts[i]/np.sum(counts)
        entropy -= prob * math.log2(prob)
    return entropy


def visualize(X1, X2, c1, c2):
    X_embedded1 = TSNE(n_components=2).fit_transform(X1)
    X_embedded2 = TSNE(n_components=2).fit_transform(X2)
    # plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=50, color=color)
    # plt.xlabel("x")
    # plt.ylabel("y")
    fig, ax = plt.subplots()
    ax.scatter(X_embedded1[:, 0], X_embedded1[:, 1], s=50, color=c1)
    ax.scatter(X_embedded2[:, 0], X_embedded2[:, 1], s=50, color=c2)


def visualize_1(X, Y):
    # X_embedded = TSNE(n_components=2).fit_transform(X)
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], s=50, c=Y)