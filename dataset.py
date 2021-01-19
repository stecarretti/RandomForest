import numpy as np


def gaussians_dataset(n_gaussian, n_points, mus, stds):
    """
    Provides a dataset made by several gaussians.

    Parameters
    ----------
    n_gaussian : int
        The number of desired gaussian components.
    n_points : list
        A list of cardinality of points (one for each gaussian).
    mus : list
        A list of means (one for each gaussian, e.g. [[1, 1], [3, 1]).
    stds : list
        A list of stds (one for each gaussian, e.g. [[1, 1], [2, 2]).

    Returns
    -------
    tuple
        a tuple like:
            data ndarray shape: (n_samples, dims).
            class ndarray shape: (n_samples,).
    """

    assert n_gaussian == len(mus) == len(stds) == len(n_points)

    data = []
    cl = []
    for i in range(0, n_gaussian):

        mu = mus[i]
        std = stds[i]
        n_pt = n_points[i]

        cov = np.diag(std)

        data.append(np.random.multivariate_normal(mu, cov, size=n_pt))
        cl.append(np.ones(shape=n_pt) * i)

    data = np.concatenate(data, axis=0)
    cl = np.concatenate(cl, axis=0)

    return data, cl