import numpy as np


def get_explained_variance_ratio(singular_value_matrix: np.ndarray) -> np.ndarray:
    """
    Used to calculate the energy according to principal components

    Args:
        singular_value_matrix (np.ndarray): Diagonal singular value matrix from SVD

    Returns:
        np.ndarray: explained variance ratio ("Energy") per principle component

    """
    explained_variance_ratio = singular_value_matrix ** 2 / np.sum(singular_value_matrix ** 2)
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

    if not get_full_reconstruction:
        S[n_principal_components:] = 0

    reconstructed_dataset_centered = U.dot(np.diag(S).dot(Vh))
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

    _, svd_matrices = get_pca_reconstruction(
        dataset=image, get_full_reconstruction=True, n_principal_components=n_principal_components
    )
    S = svd_matrices[1]

    explained_variance = get_explained_variance_ratio(S)

    # calculate energy for every principle component
    for principle_component in range(n_principal_components):

        explained_variance_ratio = np.sum(explained_variance[principle_component:])
        lost_energy.append(explained_variance_ratio)

    # find first index under threshold
    lost_energy_np = np.array(lost_energy)
    all_threshold_pass_idx = np.where(lost_energy_np < truncation_error_threshold)
    first_threshold_pass_idx = np.min(all_threshold_pass_idx)

    return lost_energy_np, first_threshold_pass_idx
