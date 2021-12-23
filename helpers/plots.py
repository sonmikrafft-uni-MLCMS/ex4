# Module to plot a dataset

import matplotlib
import matplotlib.pyplot as plt
from helpers.pca import get_explained_variance_ratio
import numpy as np
import math

from typing import Optional

matplotlib.rcParams["figure.figsize"] = (7, 5)
CMAP = plt.cm.Spectral
plt.set_cmap(CMAP)
plt.close()


def plot_2d_dataset(
    x: list[float],
    y: list[float],
    x_label: str,
    y_label: str,
    title: str,
    xlim: Optional[list[float]] = None,
    ylim: Optional[list[float]] = None,
    axis_equal: bool = False,
    colorbar: bool = False,
    colorbar_label: str = "",
    **kwargs,
) -> tuple[plt.Axes, plt.Figure]:
    """[summary]

    Args:
        x (list[float]): List of data to plot on x-axis.
        y (list[float]): List of data to plot on y-axis.
        x_label (str): Label for x-axis.
        y_label (str): Label for y-axis.
        title (str): Title of the plot.
        xlim (Optional[list[float, float]], optional): Limits for x-axis. Defaults to None.
        ylim (Optional[list[float, float]], optional): Limits for y-axis. Defaults to None.
        axis_equal (book, optional): Whether to have equal axis. Defaults to False.
        colorbar (bool, optional): Whether to show a colorbar. Defaults to False.
        colorbar_label (str, optional): Label for the colorbar. Defaults to "".
        **kwargs: Parameters for matplotlib.pyplot.scatter.

    Returns:
        tuple[plt.Axes, plt.Figure]: Tuple of axes and figure.
    """
    fig, ax = plt.subplots()
    ax.scatter(x, y, **kwargs)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if axis_equal:
        ax.axis("equal")
    ax.grid()
    if colorbar:
        cbar = fig.colorbar(ax.collections[0])
        cbar.set_label(colorbar_label)
    return fig, ax


def plot_3d_dataset(
    x: list[float],
    y: list[float],
    z: list[float],
    x_label: str,
    y_label: str,
    z_label: str,
    title: str,
    axis_equal: bool = False,
    colorbar: bool = False,
    colorbar_label: str = "",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """[summary]

    Args:
        x (list[float]): List of data to plot on x-axis.
        y (list[float]): List of data to plot on y-axis.
        z (list[float]): List of data to plot on z-axis.
        x_label (str): Label for x-axis.
        y_label (str): Label for y-axis.
        z_label (str): Label for z-axis.
        title (str): Title of the plot.
        axis_equal (book, optional): Whether to have equal axis. Defaults to False.
        colorbar (bool, optional): Whether to show a colorbar. Defaults to False.
        colorbar_label (str, optional): Label for the colorbar. Defaults to "".
        **kwargs: Parameters for matplotlib.pyplot.scatter.

    Returns:
        tuple[plt.Figure, plt.Axes]: Tuple of figure and axes.
    """
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x, y, z, **kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    if axis_equal:
        ax.axis("equal")
    if colorbar:
        cbar = fig.colorbar(ax.collections[0])
        cbar.set_label(colorbar_label)
    fig.tight_layout()
    return fig, ax


def plot_two_pedestrians(
    X_pedestrians: np.ndarray, x_label: str, y_label: str, title: str, save_path: Optional[str] = None
) -> None:
    """Given the dataset of the pedestrians, plot the first two pedestrians.

    Args:
        X_pedestrians (np.ndarray): Dataset of the pedestrians.
        x_label (str): [description]: Label for x-axis.
        y_label (str): [description]: Label for y-axis.
        title (str): [description]: Title of the plot.
        save_path (Optional[str], optional): [description]. Defaults to None.
    """

    X1, X2 = X_pedestrians[:, :2], X_pedestrians[:, 2:4]

    fig, ax = plt.subplots()
    ax.scatter(X1[:, 0], X1[:, 1], marker="o", label="Pedestrian 1")
    ax.scatter(X2[:, 0], X2[:, 1], marker="o", label="Pedestrian 2")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.axis("equal")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    fig.show()


def plot_pca_transformation(transformed_data_1d: np.ndarray, transformed_data_2d: np.ndarray) -> None:
    """Plots a two dimensional dataset and the corresponding linear 1D subspace

    Args:
        transformed_data_1d (np.ndarray): linear 1D subspace
        transformed_data_2d (np.ndarray): two dimensional dataset
    """
    principal_component_1 = transformed_data_2d[:, 0]
    principal_component_2 = transformed_data_2d[:, 1]

    principal_component_subspace_1 = transformed_data_1d[:, 0]

    if transformed_data_1d.shape[1] == 1:
        principal_component_subspace_2 = np.zeros_like(principal_component_subspace_1)
    else:
        principal_component_subspace_2 = transformed_data_1d[:, 1]

    plt.figure(figsize=(15, 10))
    plt.scatter(principal_component_1, principal_component_2, label=" transformed samples", alpha=0.5)
    plt.scatter(principal_component_subspace_1, principal_component_subspace_2, label="1D transformed subspace")
    plt.axis("equal")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Transformed PCA Dataset")
    plt.legend()


def plot_principal_components(reconstructed_dataset: np.ndarray, svd_matrices: tuple, scale_components: bool) -> None:
    """Plots the points dataset and indicates corresponding principle components with terse lines fixed to origin

    Args:
        reconstructed_dataset (np.ndarray): pca dataset reconstruction using SVD
        svd_matrices (tuple): eigendecomposition matrices used during SVD for example for explained variance analysis
        scale_components (bool): flag to set length of principle component terse line based on variance

    Adapted from:https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html#sphx-glr-\
        auto-examples-cross-decomposition-plot-pcr-vs-pls-py
    """

    S = svd_matrices[1]
    Vh = svd_matrices[2]

    x = reconstructed_dataset[:, 0]
    f_x = reconstructed_dataset[:, 1]

    var_explained = get_explained_variance_ratio(S)

    plt.figure(figsize=(15, 10))
    plt.scatter(x, f_x, label="samples")

    for i, (comp, var) in enumerate(zip(Vh, var_explained)):
        if scale_components:
            comp = comp * np.sqrt(var) * 2
        plt.plot(
            [0, comp[0]],
            [0, comp[1]],
            label=f"Component {i+1}",
            linewidth=3,
            color=f"C{i + 2}",
        )

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("PCA Dataset")
    plt.legend()


def plot_image_reconstruction(reconstructed_images: list, n_principal_components: list) -> None:
    """Plots the reconstructed images based on different numbers of principle components

    Args:
        reconstructed_images (list): reconstructed racoon images for different numbers of principle components
        n_principal_components (int): number of principle components used for reconstruction
    """

    fig, ax = plt.subplots(2, 2, figsize=(20, 17))

    plt.suptitle("Comparison Image Reconstruction", fontsize=20, y=1)
    fig.tight_layout()
    ax[0, 0].imshow(reconstructed_images[0], cmap="gray")
    ax[0, 0].set_title(f"Number of principle components = {n_principal_components[0]}")
    ax[0, 1].imshow(reconstructed_images[1], cmap="gray")
    ax[0, 1].set_title(f"Number of principle components = {n_principal_components[1]}")
    ax[1, 0].imshow(reconstructed_images[2], cmap="gray")
    ax[1, 0].set_title(f"Number of principle components = {n_principal_components[2]}")
    ax[1, 1].imshow(reconstructed_images[3], cmap="gray")
    ax[1, 1].set_title(f"Number of principle components = {n_principal_components[3]}")


def plot_truncation_error(lost_energy: np.ndarray, truncation_error_threshold: float, zoom: bool) -> None:
    """Create a bar chart for every principle component illustrating the explained variance

    Args:
        lost_energy (np.array): contains sum of explained variance ("energy") for each principle component
        truncation_error_threshold (float): [description]
        zoom (bool): [description]
    """
    xwerte = np.arange(0, lost_energy.shape[0], 1)

    plt.figure(figsize=(15, 10))
    plt.bar(xwerte, lost_energy, label="Energy in principal component")
    plt.axhline(y=truncation_error_threshold, color="r", linestyle="-", label="Truncation error")
    plt.xlabel("Number of principle components")
    if zoom:
        plt.ylim([0, 0.12])
        plt.xlim([0, 100])
    plt.ylabel("Lost Explained Variance")
    plt.legend()
    plt.show()


def plot_pedestrian_trajectories(dataset: np.ndarray) -> None:
    """Plots the first two trajectories of the DMAP PCA Vadere dataset

    Args:
        dataset (np.ndarray): Dataset with pedestrian trajectories encoded in two columns over 1000 timestamps
    """

    x_1 = dataset[:, 0]
    y_1 = dataset[:, 1]
    x_2 = dataset[:, 2]
    y_2 = dataset[:, 3]

    plt.figure(figsize=(15, 10))

    plt.plot(x_1, y_1, label="Trajectory Pedestrian 1")
    plt.plot(x_2, y_2, label="Trajectory Pedestrian 2")

    plt.plot(x_1[0], y_1[0], "ro", label="Pedestrian 1 - Startpoint")
    plt.plot(x_1[-1], y_1[-1], "go", label="Pedestrian 1 - Endpoint")

    plt.plot(x_2[0], y_2[0], "bo", label="Pedestrian 2 - Startpoint")
    plt.plot(x_2[-1], y_2[-1], "ko", label="Pedestrian 2 - Endpoint")

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.title("DMAP PCA Vadere Dataset")
    plt.legend()


def plot_projection_subspace(
    dataset: np.ndarray, reconstructed_dataset: np.ndarray, svd_matrices: tuple, scale_components: bool, transform: bool
) -> None:
    """
    Plots original dataset and approximates the corresponding reduced dimensionality subspace in original/transformed
    coordinate frames.

    Args:
        dataset (np.ndarray): Original PCA dataset
        reconstructed_dataset (np.ndarray): Original reconstructed PCA dataset (= pca dataset with dimension reduced to
            1D)
        svd_matrices (tuple): eigendecomposition matrices used during SVD for example for explained variance analysis
        scale_components (bool): flag to set length of principle component terse line based on variance
        transform (bool): flag to transform dataset to principle component coordinate frame
    """
    S = svd_matrices[1]
    V = svd_matrices[2]

    x_label = "x"
    y_label = "f(x)"

    show_principle_components = True

    if transform:
        dataset = dataset.dot(V.T)
        reconstructed_dataset = reconstructed_dataset.dot(V.T)
        x_label = "Principal Component 1"
        y_label = "Principal Component 2"
        show_principle_components = False

    x = dataset[:, 0]
    f_x = dataset[:, 1]
    x_reconstructed = reconstructed_dataset[:, 0]
    f_x_reconstructed = reconstructed_dataset[:, 1]

    var_explained = get_explained_variance_ratio(S)

    plt.figure(figsize=(15, 10))
    plt.scatter(x, f_x, label="2D samples", alpha=0.4)
    plt.scatter(x_reconstructed, f_x_reconstructed, label="1D Subspace")

    if show_principle_components:
        for i, (comp, var) in enumerate(zip(V, var_explained)):
            if scale_components:
                comp = comp * np.sqrt(var) * 2
            plt.plot(
                [0, comp[0]],
                [0, comp[1]],
                label=f"Component {i+1}",
                linewidth=3,
                color=f"C{i + 2}",
            )

    plt.axis("equal")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("PCA Dataset")
    plt.legend()


def plot_trajectory_2D_subspace(svd_matrices: tuple) -> None:
    """Projects the 30 dimensional dataset to the first two principle components

    Args:
        svd_matrices (tuple): eigendecomposition matrices used during SVD for example for explained variance analysis
    """

    S = svd_matrices[1]
    U = svd_matrices[0]

    Dmap_transformed = U * S

    PC1 = Dmap_transformed[:, 0]
    PC2 = Dmap_transformed[:, 1]

    fig, ax = plt.subplots(figsize=(17, 12))
    col = [plt.cm.jet(float(i) / (len(PC1))) for i in range(len(PC1))]

    ax.scatter(PC1, PC2, c=col, label="2D Subspace")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Projection 2D Subspace")
    cbar = fig.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Normalized Timestamp Factor=1000")
    ax.legend()


def plot_trajectory_3D_subspace(svd_matrices):
    """Projects the 30 dimensional dataset to the first three principle components

    Args:
        svd_matrices (tuple): eigendecomposition matrices used during SVD for example for explained variance analysis
    """

    S = svd_matrices[1]
    U = svd_matrices[0]
    Dmap_transformed = U * S

    PC1 = Dmap_transformed[:, 0]
    PC2 = Dmap_transformed[:, 1]
    PC3 = Dmap_transformed[:, 2]

    fig = plt.figure(figsize=(17, 12))
    ax = fig.add_subplot(projection="3d")

    col = [plt.cm.jet(float(i) / (len(PC1))) for i in range(len(PC1))]
    ax.scatter(PC1, PC2, PC3, c=col, label="3D Subspace")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Projection 3D Subspace")
    ax.view_init(elev=45, azim=40)
    cbar = fig.colorbar(ax.collections[0])
    cbar.set_label("Normalized Timestamp Factor=1000")
    fig.tight_layout()
    ax.legend()
    plt.show()


def plot_pairwise_eigenvector(
    eigenvectors: np.ndarray,
    n_cols: int = 2,
    individual_size: tuple[int, int] = (7, 3),
    axis_equal: bool = False,
    xlim: Optional[list[float]] = None,
    ylim: Optional[list[float]] = None,
    save_path: Optional[str] = None,
    title_suffix: str = "",
    **kwargs,
) -> None:
    """Plot pairwise eigenvector.

    Args:
        eigenvectors (np.ndarray): Array of eigenvectors of shape (n_samples, n_eigenvectors).
        n_cols (int, optional): Number of columns. Defaults to 2.
        individual_size (tuple[int, int], optional): Size of the individual plot. Defaults to (7, 5).
        axis_equal (bool, optional): Whether to have equal axis. Defaults to False.
        xlim (Optional[list[float]], optional): Limits for x-axis. Defaults to None.
        ylim (Optional[list[float]], optional): Limits for y-axis. Defaults to None.
        save_path (Optional[str], optional): Path to save. Defaults to None.
        title_suffix (str, optional): Suffix for the title. Defaults to "".
        **kwargs: Parameters for matplotlib.pyplot.scatter.
    """
    num_eigenvectors = eigenvectors.shape[1] - 1
    nrows = int(math.ceil(num_eigenvectors / n_cols))
    fig, axs = plt.subplots(nrows, n_cols, figsize=(n_cols * individual_size[0], nrows * individual_size[1]))

    correction = 0
    for i in range(num_eigenvectors):
        if i == 1:
            correction = 1
        ax = axs.flatten()[i]
        ax.scatter(eigenvectors[:, 1], eigenvectors[:, i + correction], **kwargs)
        ax.set_title(r"$\psi_1$ vs. $\psi_{}${}".format(i + correction, title_suffix))
        ax.grid()
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if axis_equal:
            ax.axis("equal")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")


def plot_3d_pca_plot(
    X: np.ndarray,
    dir_axis: np.ndarray,
    x_label: str,
    y_label: str,
    z_label: str,
    title: str,
    add_mean_to_axis: bool = True,
    save_path: Optional[str] = None,
    **kwargs,
) -> None:
    """Plot 3D PCA plot.

    Args:
        X (ndarray): 3D data of shape (n_samples, 3).
        dir_axis (ndarray): Direction of coordinate axis.
            dir_axis[0] is the direction of x-axis,
            dir_axis[1] is the direction of y-axis,
            dir_axis[2] is the direction of z-axis.
        x_label (str): Label for x-axis.
        y_label (str): Label for y-axis.
        z_label (str): Label for z-axis.
        title (str): Title of the plot.
        add_mean_to_axis (bool, optional): Whether to add mean to axis. Defaults to True.
        save_path (Optional[str], optional): Path to save. Defaults to None.
        **kwargs: Parameters for matplotlib.pyplot.scatter of the data.
    """
    X_mean = np.zeros(3)
    if add_mean_to_axis:
        X_mean = np.mean(X, axis=0)

    fig, ax = plot_3d_dataset(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        x_label=x_label,
        y_label=y_label,
        z_label=z_label,
        title=title,
        **kwargs,
    )

    scale = 20
    colors = ["red", "green", "blue"]
    labels = ["PC1", "PC2", "PC3"]
    for dir, color, label in zip(dir_axis, colors, labels):
        ax.plot(
            [X_mean[0], X_mean[0] + scale * dir[0]],
            [X_mean[1], X_mean[1] + scale * dir[1]],
            [X_mean[2], X_mean[2] + scale * dir[2]],
            color=color,
            lw=3,
            label=label,
        )
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")


def plot_2d_pca_plot(
    X: np.ndarray,
    dir_axis: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    add_mean_to_axis: bool = True,
    save_path: Optional[str] = None,
    **kwargs,
) -> None:
    """Plot 2D PCA plot.

    Args:
        X (ndarray): 2D data of shape (n_samples, 2).
        dir_axis (ndarray): Direction of coordinate axis.
            dir_axis[0] is the direction of x-axis,
            dir_axis[1] is the direction of y-axis,
            dir_axis[2] is the direction of z-axis.
        x_label (str): Label for x-axis.
        y_label (str): Label for y-axis.
        title (str): Title of the plot.
        add_mean_to_axis (bool, optional): Whether to add mean to axis. Defaults to True.
        save_path (Optional[str], optional): Path to save. Defaults to None.
        **kwargs: Parameters for matplotlib.pyplot.scatter of the data.
    """
    X_mean = np.zeros(3)
    if add_mean_to_axis:
        X_mean = np.mean(X, axis=0)

    fig, ax = plot_2d_dataset(
        X[:, 0],
        X[:, 1],
        x_label=x_label,
        y_label=y_label,
        title=title,
        **kwargs,
    )

    scale = 20
    colors = ["red", "green"]
    labels = ["PC1", "PC2"]
    for dir, color, label in zip(dir_axis, colors, labels):
        ax.plot(
            [X_mean[0], X_mean[0] + scale * dir[0]],
            [X_mean[1], X_mean[1] + scale * dir[1]],
            color=color,
            lw=3,
            label=label,
        )
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")


def plot_2d_train_test(
    train: np.ndarray, test: np.ndarray, xlabel: str, ylabel: str, title: str, save_path: Optional[str] = None, **kwargs
) -> None:
    """Plot train and test data of 2D dataset.

    Args:
        train (ndarray): 2D data of shape (n_samples, 2).
        test (ndarray): 2D data of shape (n_samples, 2).
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        title (str): Title of the plot.
        save_path (Optional[str], optional): Path to save. Defaults to None.
        **kwargs: Parameters for matplotlib.pyplot.scatter of both splits.
    """
    fig, ax = plt.subplots()
    ax.scatter(train[:, 0], train[:, 1], color="red", label="train", marker="o", s=6, **kwargs)
    ax.scatter(test[:, 0], test[:, 1], color="blue", label="test", marker="^", s=2, **kwargs)
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
