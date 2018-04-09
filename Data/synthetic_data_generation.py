from sklearn.datasets import make_blobs


def syn_data_points(n_samples=10000):
    """
    returns 2-Dimensional data points centers around the given centers

    :param n_samples:
    :return:
    """
    # centers of the clusters
    centers = [(5, 5), (-5, 5), (-5, -5), (5, -5)]

    x, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                      centers=centers, shuffle=False, random_state=42)

    return x, y


if __name__ == '__main__':
    syn_data_points()
