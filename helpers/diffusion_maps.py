# Module to implement the diffusion map algorithm.

import numpy as np

from scipy.sparse.linalg import eigsh
from scipy.linalg import fractional_matrix_power
from scipy.spatial import distance_matrix

np.random.seed(66)


def get_epsilon(distance_matrix: np.ndarray) -> float:
    """
    Given a distance matrix of shape (n, n), returns the epsilon value for the diffusion map.
    Set epsilon to 5% of the diameter of the dataset.

    Args:
        distance_matrix (np.ndarray): A distance matrix of shape (n, n).

    Returns:
        float: The epsilon value for the diffusion map.
    """
    return 0.05 * np.max(distance_matrix)


def get_kernel_maxtrix(distance_matrix: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Given a distance matrix of shape (n, n), returns a kernel matrix of shape (n, n) where each entry (i, j) is the
    kernel value between data[i] and data[j].

    Args:
        distance_matrix (np.ndarray): A distance matrix of shape (n, n).
        epsilon (float): The epsilon value for the diffusion map.

    Returns:
        np.ndarray: A kernel matrix of shape (n, n).
    """
    w = np.exp(-(distance_matrix ** 2) / epsilon)
    return w


def get_diagonal_normalization_matrix(kernel_matrix: np.ndarray) -> np.ndarray:
    """
    Given a kernel matrix of shape (n, n), returns a diagonal normalization matrix of shape (n, n) where each
    entry (i, i) on the diagonal is the sum over the columns at the ith row of the kernel matrix.

    Args:
        kernel_matrix (np.ndarray): A kernel matrix of shape (n, n).

    Returns:
        np.ndarray: A diagonal normalization matrix of shape (n, n).
    """
    p = np.diag(np.sum(kernel_matrix, axis=1))
    return p


def get_normalized_kernel_matrix(kernel_matrix: np.ndarray, diagonal_normalization_matrix: np.ndarray) -> np.ndarray:
    """
    Given a kernel matrix of shape (n, n) and its diagonal normalization matrix of shape (n, n),
    returns the normalized kernel matrix of shape (n, n).

    Args:
        kernel_matrix (np.ndarray): A kernel matrix of shape (n, n).
        diagonal_normalization_matrix (np.ndarray): A diagonal normalization matrix of shape (n, n) that is the
            normalization of the kernel matrix.

    Returns:
        np.ndarray: A normalized kernel matrix of shape (n, n).
    """
    return np.linalg.inv(diagonal_normalization_matrix) @ kernel_matrix @ np.linalg.inv(diagonal_normalization_matrix)


def get_diffusion_map(data_matrix: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Given a dataset of shape (n, d), applies a diffusion map algorithm to the dataset and returns the embedding along
    the first L eigenfunction associated with the largest eigenvalues, and the eigenvalues.

    This function is based on the implementation of the diffusion map algorithm in the paper:
    https://epubs.siam.org/doi/pdf/10.1137/12088183X

    It tries to remove the influence of the sampling density of the data points by using some clever normalization.

    Args:
        data_matrix (np.ndarray): A dataset of shape (n, d).
        L (int): The number of eigenfunctions to return. This includes the zero eigenfunction, so a value of L results
            in L+1 eigenfunctions.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays. The first array is the embedding of the dataset
            along the first L eigenfunctions associated with the largest eigenvalues, and the second array is the
            eigenvalues.

            The embedding is of shape (n, L+1), and the eigenvalues are of shape (L+1,).
    """
    # form kernel matrix
    D = distance_matrix(data_matrix, data_matrix)
    epsilon = get_epsilon(D)
    W = get_kernel_maxtrix(D, epsilon)
    
    # normalization
    P = get_diagonal_normalization_matrix(W)
    K = get_normalized_kernel_matrix(W, P)
    Q = get_diagonal_normalization_matrix(K)
    frac = fractional_matrix_power(Q, -0.5)
    T_hat = frac @ K @ frac

    # eigendecomposition
    eig_val, eig_vec = eigsh(T_hat, k=L + 1)  # use eigsh (vs. eigs) since matrix is symmetric
    eig_val, eig_vec = np.flip(eig_val, axis=0), np.flip(eig_vec, axis=1)

    # return embedding and eigenvalues
    return np.sqrt(eig_val ** (1 / epsilon)), frac @ eig_vec
