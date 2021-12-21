import math
import scipy.misc
import numpy as np
import pandas as pd
from sklearn.datasets import make_swiss_roll
from skimage.transform import resize
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


def get_racoon_img():
    """ 
    Used to create and resize an sample image of a racoon

    Returns:
        racoon_img_reshaped (np.ndarray): Resized image of racoon
    """
    racoon_img = scipy.misc.face(gray=True)
    racoon_img_reshaped = resize(racoon_img, (249, 185))
    return racoon_img_reshaped


def get_explained_variance_ratio(singular_value_matrix: np.ndarray) -> np.ndarray:
    """ 
    Used to calculate the energy according to principal components

    Args:
        singular_value_matrix (np.ndarray): Diagonal singular value matrix from SVD

    Returns:
        np.ndarray: explained variance ratio ("Energy") per principle component

    """
    explained_variance_ratio = singular_value_matrix**2/np.sum(singular_value_matrix**2)
    return explained_variance_ratio


def get_pca_reconstruction(dataset: np.ndarray, get_full_reconstruction: bool, n_principal_components: int):
    """ 
    Reconstructs or reduces dimensions of a given dataset by eigendecomposition of the data matrix

    Args:
        dataset (np.ndarray): provided example dataset containing (points, images or trajectories)
        get_full_reconstruction (bool): flag to initialize dimension reduction on dataset
        n_principal_components (int): number of principle components used for reconstruction

    Returns:
        reconstructed_dataset (np.ndarray): dimensional reduced reconstruction of the given dataset
        svd_matrices (tuple): eigendecomposition matrices used during SVD for example for explained variance analyses
    """
    mean = np.mean(dataset, axis=0) 
    dataset_centered = dataset - mean 
    
    U, S, Vh = np.linalg.svd(dataset_centered, full_matrices=False)
    
    if get_full_reconstruction == False:
        S[n_principal_components:] = 0

    reconstructed_dataset_centered= U.dot(np.diag(S).dot(Vh))
    reconstructed_dataset = reconstructed_dataset_centered + mean 
    svd_matrices = (U, S, Vh)
    
    return reconstructed_dataset, svd_matrices


def get_truncation_error(truncation_error_threshold: float, image: np.ndarray, n_principal_components: int):
    """
    Calculates the explained variance through truncation ("lost energy") for all principle components

    Args:
        truncation_error_threshold: (float): threshold for truncation error analysis 
        image (np.ndarray): dataset used for energy analysis (racoon image)
        n_principal_components (int): number of principle components 
    Returns:
        lost_energy (np.ndarray): array containing sum of explained variances ("energy") for every principle components 
    """

    lost_energy = []

    _, svd_matrices = get_pca_reconstruction(dataset=image,get_full_reconstruction=True, n_principal_components=n_principal_components)
    S = svd_matrices[1]

    explained_variance = get_explained_variance_ratio(S)

    #calculate energy for every principle component 
    for principle_component in range(n_principal_components):

        explained_variance_ratio = np.sum(explained_variance[principle_component:])
        lost_energy.append(explained_variance_ratio)
    
    #find first index under threshold
    lost_energy = np.array(lost_energy)
    all_threshold_pass_idx = np.where(lost_energy < truncation_error_threshold)
    first_threshold_pass_idx = np.min(all_threshold_pass_idx)

    return lost_energy, first_threshold_pass_idx