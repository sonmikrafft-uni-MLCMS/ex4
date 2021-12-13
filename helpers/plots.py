from typing import Optional
import matplotlib.pyplot as plt
from numpy import ndarray
import numpy as np

# collection of functions to plot a dataset

CMAP = plt.cm.Spectral
plt.set_cmap(CMAP)
plt.close()


def plot_2d_dataset(
    x: list[float],
    y: list[float],
    x_label: str,
    y_label: str,
    title: str,
    save_path: Optional[str] = None,
    c: Optional[str] = None,
) -> None:
    """[summary]

    Args:
        x (list[float]): List of data to plot on x-axis.
        y (list[float]): List of data to plot on y-axis.
        x_label (str): Label for x-axis.
        y_label (str): Label for y-axis.
        title (str): Title of the plot.
        save_path (Optional[str], optional): Path to save. Defaults to None.
        c (Optional[str], optional): Color of the points. Defaults to None.
    """
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=c)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.axis("equal")
    ax.grid()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    fig.show()


def plot_3d_dataset(
    x: list[float],
    y: list[float],
    z: list[float],
    x_label: str,
    y_label: str,
    z_label: str,
    title: str,
    save_path: Optional[str] = None,
    c: Optional[str] = None,
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
        save_path (Optional[str], optional): Path to save. Defaults to None.
        c (Optional[str], optional): Color of the points. Defaults to None.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x, y, z, c=c)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    fig.show()


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
    fig.show()


def plot_pca_transformation(transformed_data_1d: np.ndarray, transformed_data_2d: np.ndarray) -> None:
    """ Plots a two dimensional dataset and the corresponding linear 1D subspace

    Args:
        transformed_data_1d (np.ndarray): linear 1D subspace
        transformed_data_2d (np.ndarray): two dimensional dataset
    """
    principal_component_1 = transformed_data_2d[:,0]
    principal_component_2 = transformed_data_2d[:,1]  

    principal_component_subspace_1 = transformed_data_1d[:,0]
    
    if transformed_data_1d.shape[1] == 1:
        principal_component_subspace_2 = np.zeros_like(principal_component_subspace_1)
    else:
        principal_component_subspace_2 = transformed_data_1d[:,1]
    
    plt.figure(figsize=(15,10))
    plt.scatter(principal_component_1, principal_component_2, label=" transformed samples", alpha=0.5)
    plt.scatter(principal_component_subspace_1, principal_component_subspace_2, label="1D transformed subspace")
    plt.axis('equal')
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Transformed PCA Dataset")
    plt.legend()

def plot_principal_components(dataset: np.ndarray, model, scale_components: bool) -> None:
    """Plots a two dimensional dataset and the corresponding principal components

    Args:
        dataset (np.ndarray): two dimensional dataset
        model (pca object): pca model containing the principal components, variance and energy
        scale_components (bool): flag to scale principal components vectors in dependence of their variance

    Source: https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html#sphx-glr-auto-examples-cross-decomposition-plot-pcr-vs-pls-py

    """
    x = dataset[:,0]
    f_x = dataset[:,1]

    plt.figure(figsize=(15,10))
    plt.scatter(x, f_x, label="samples")

    for i, (comp, var) in enumerate(zip(model.components_, model.explained_variance_)):
        if scale_components == True:
            comp = comp * np.sqrt(var) * 2
        plt.plot(
            [0, comp[0]],
            [0, comp[1]],
            label=f"Component {i+1}",
            linewidth=3,
            color=f"C{i + 2}",
        )

    plt.axis('equal')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("PCA Dataset")
    plt.legend()