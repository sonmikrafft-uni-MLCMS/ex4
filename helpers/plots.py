import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

from typing import Optional

# collection of functions to plot a dataset

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


def plot_pairwise_eigenvector(
    eigenvectors: np.ndarray,
    axis_equal: bool = False,
    xlim: Optional[list[float]] = None,
    ylim: Optional[list[float]] = None,
    save_path: Optional[str] = None,
    **kwargs,
) -> None:
    """Plot pairwise eigenvector.

    Args:
        eigenvectors (np.ndarray): Array of eigenvectors of shape (n_samples, n_eigenvectors).
        axis_equal (bool, optional): Whether to have equal axis. Defaults to False.
        xlim (Optional[list[float]], optional): Limits for x-axis. Defaults to None.
        ylim (Optional[list[float]], optional): Limits for y-axis. Defaults to None.
        save_path (Optional[str], optional): Path to save. Defaults to None.
        **kwargs: Parameters for matplotlib.pyplot.scatter.
    """
    num_eigenvectors = eigenvectors.shape[1] - 1
    ncols = 2
    nrows = int(math.ceil(num_eigenvectors / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5))

    correction = 0
    for i in range(num_eigenvectors):
        if i == 1:
            correction = 1
        ax = axs.flatten()[i]
        ax.scatter(eigenvectors[:, 1], eigenvectors[:, i + correction], **kwargs)
        ax.set_title(r"$\psi_1$ vs. $\psi_{}$".format(i + correction))
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
