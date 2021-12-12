import math

import numpy as np
import pandas as pd
from sklearn.datasets import make_swiss_roll

# collection of functions to generate datasets


def _get_unit_cycle_data_k(k: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Helper function to generate uniformly distributed points on the unit circle.

    Calculates the k-th term of the sequence of points on the unit circle as:
    t = (2 * np.pi * k) / (N + 1)
    x = (cos(t), sin(t))

    Args:
        k (int): Current index of the point to generate.
        N (int): Total number of points to generate.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of t and x.
    """
    t = np.array(2 * math.pi * k / (N + 1))
    x = np.array([np.cos(t), np.sin(t)])
    return x, t


def get_unit_cycle_dataset(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a dataset of N points, uniformly distributed on the unit circle.

    The data is defined as the set of all 2D points x in R^2, for k = 1, ..., N, as:
        x = (cos(t), sin(t))
        t = (2 * np.pi * k) / (N + 1)

    Args:
        N (int): Number of points in the dataset to generate uniformly distributed on the unit circle.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of data, t, and colors.
            Data is a 2D array of shape (N, 2), where each row is a point in the dataset.
            Colors is a 1D array of shape (N,), where each element is the color of the corresponding point.
    """
    k_vec = np.arange(1, N + 1)
    colors = k_vec
    x, t  = _get_unit_cycle_data_k(k_vec, len(k_vec))
    return x.T, t, colors


def get_swiss_roll_dataset(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Get swiss roll dataset of N points.

    Getting dataset from https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html.

    The data is defined as the set of all 3D points x in R^3, for k = 1, ..., N, as:
        x = (u*cos(u), v, u*sin(u))
        where (u, v) in [0, 10]^2 are chosen uniformly at random.

    Args:
        N (int): N points in the dataset to generate.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of data and colors.
            Data is a 3D array of shape (N, 3), where each row is a point in the dataset.
            Colors is a 1D array of shape (N,), where each element is the color of the corresponding point.
    """

    x, color = make_swiss_roll(n_samples=N)
    return x, color


def get_trajectory_dataset(path: str = "data/data_DMAP_PCA_vadere.txt") -> np.ndarray:
    """
    Get the trajectory dataset exported from Vadere.
    Dataset contains position data of 15 pedestrians over 1000 timesteps.
    Data in the form of a 2D array of shape (1000, 30).
    Each row contains positions of 15 pedestrians in the form of (x1, y1, x2, y2, ..., x15, y15).

    Args:
        path (str, optional): Path to the data file. Defaults to "data/data_DMAP_PCA_vadere.txt".

    Returns:
        np.ndarray: Data in the form of a 2D array of shape (1000, 30).
            Each row contains positions of 15 pedestrians in the form of (x1, y1, x2, y2, ..., x15, y15).

    """
    return pd.read_csv(path, delimiter=" ", header=None).to_numpy()
