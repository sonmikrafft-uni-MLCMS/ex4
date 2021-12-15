from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
from numpy import ndarray

# collection of functions to plot a dataset

matplotlib.rcParams['figure.figsize'] = (7, 5)
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
    save_path: Optional[str] = None,
    c: Optional[str] = None,
    colorbar: bool = False,
    colorbar_label: str = "",
) -> None:
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
        save_path (Optional[str], optional): Path to save. Defaults to None.
        c (Optional[str], optional): Color of the points. Defaults to None.
        colorbar (bool, optional): Whether to show a colorbar. Defaults to False.
        colorbar_label (str, optional): Label for the colorbar. Defaults to "".
    """
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=c)
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
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)


def plot_3d_dataset(
    x: list[float],
    y: list[float],
    z: list[float],
    x_label: str,
    y_label: str,
    z_label: str,
    title: str,
    axis_equal: bool = False,
    save_path: Optional[str] = None,
    c: Optional[str] = None,
    colorbar: bool = False,
    colorbar_label: str = "",
) -> None:
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
        save_path (Optional[str], optional): Path to save. Defaults to None.
        c (Optional[str], optional): Color of the points. Defaults to None.
        colorbar (bool, optional): Whether to show a colorbar. Defaults to False.
        colorbar_label (str, optional): Label for the colorbar. Defaults to "".
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x, y, z, c=c)
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
    if save_path:
        fig.savefig(save_path)


def plot_two_pedestrians(
    X_pedestrians: ndarray, x_label: str, y_label: str, title: str, save_path: Optional[str] = None
) -> None:
    """[summary]

    Args:
        X_pedestrians (ndarray): [description]
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
        fig.savefig(save_path)
