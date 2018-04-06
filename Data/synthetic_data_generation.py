import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def syn_data_points(n_samples=10000, centers=[(-5, -5), (5, -5), (-5, 5), (5, 5)]):
    """
    returns 2-Dimensional data points centers around the given centers
    and saves the plot

    :param n_samples:
    :param centers:
    :return:
    """
    x, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                      centers=centers, shuffle=False, random_state=42)

    # saves the plot of the synthetic data points
    plt.scatter(x[:, 0], x[:, 1], edgecolors=(0, 0, 0))
    plt.xlabel("h0")
    plt.ylabel("h1")
    plt.grid()
    plt.title("2 Dimensional Synthetic data")
    plt.savefig("../synthetic_data_points.png")

    return x, y


if __name__ == '__main__':
    syn_data_points()
