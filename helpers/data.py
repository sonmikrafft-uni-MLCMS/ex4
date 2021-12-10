import math

import numpy as np
import pandas as pd
from sklearn.datasets import make_swiss_roll

# collection of functions to generate datasets


def _get_unit_cycle_data(k: int, N: int) -> np.ndarray:
    """
    Helper function to generate uniformly distributed points on the unit circle.

    Args:
        k (int): Current index of the point to generate.
        N (int): Total number of points to generate.

    Returns:
        np.ndarray: Point on the unit circle, in the form of a 2D array of shape (2,).
    """
    t = 2 * math.pi * k / (N + 1)
    x = np.array([np.cos(t), np.sin(t)])
    return x


def get_unit_cycle_dataset(N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset of N points, uniformly distributed on the unit circle.

    Args:
        N (int): Number of points in the dataset to generate uniformly distributed on the unit circle.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of data and colors.
            Data is a 2D array of shape (N, 2), where each row is a point in the dataset.
            Colors is a 1D array of shape (N,), where each element is the color of the corresponding point.
    """
    k_vec = np.arange(1, N + 1)
    colors = k_vec
    X = _get_unit_cycle_data(k_vec, len(k_vec))
    return X.T, colors


def get_swiss_roll_dataset(N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Get swiss roll dataset of N points.

    Args:
        N (int): N points in the dataset to generate.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of data and colors.
            Data is a 3D array of shape (N, 3), where each row is a point in the dataset.
            Colors is a 1D array of shape (N,), where each element is the color of the corresponding point.
    """

    X, color = make_swiss_roll(n_samples=N)
    return X, color


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
