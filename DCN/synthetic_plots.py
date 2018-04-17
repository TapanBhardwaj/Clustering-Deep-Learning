import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF



# Assuming data of 4-class
def plot_points(x, y):
    x1 = list()
    x2 = list()
    x3 = list()
    x4 = list()
    y1 = list()
    y2 = list()
    y3 = list()
    y4 = list()
    for i in range(y.shape[0]):
        if y[i] == 1:
            x1.append(x[i, 0])
            y1.append(x[i, 1])
        elif y[i] == 2:
            x2.append(x[i, 0])
            y2.append(x[i, 1])
        elif y[i] == 3:
            x3.append(x[i, 0])
            y3.append(x[i, 1])
        else:
            x4.append(x[i, 0])
            y4.append(x[i, 1])

    plt.scatter(x1, y1, color=['red'], label='Cluster 1', edgecolors=(0, 0, 0))
    plt.scatter(x2, y2, color=['green'], label='Cluster 2', edgecolors=(0, 0, 0))
    plt.scatter(x3, y3, color=['blue'], label='Cluster 3', edgecolors=(0, 0, 0))
    plt.scatter(x4, y4, color=['orange'], label='Cluster 4', edgecolors=(0, 0, 0))
    plt.legend()
    plt.title('Plot of data points')
    plt.show()


# Assigning cluster no. based on coordinate system
def assign_cluster(x):
    if x[0] >= 0 and x[1] >= 0:
        return 1
    elif x[0] < 0 and x[1] >= 0:
        return 2
    elif x[0] < 0 and x[1] < 0:
        return 3
    else:
        return 4


# Generating 2-d points and dividing them in 4-clusters based on co-ordinate system

def sigmoid(z):
    """
    sigmoid calculation
    :param z:
    :return:
    """
    z_hat = 1 / (1 + np.exp(-z))
    return z_hat


def to_higher_dimension(x):
    """
    transform data from 2-d to 100-d using non-linear transformation
    assume data points to be 2-dimension
    x=sigmoid(U*sigmoid(W*x(in 2d)))

    :param x:
    :return:
    """
    W = np.random.normal(loc=0, scale=1, size=(100, 2))
    # U=np.random.normal(loc=0,scale=1,size=(100,10))
    x_new = np.square(sigmoid(np.dot(W, x.T))).T
    return x_new


def to_higher_dimension2(x):
    W = np.random.normal(loc=0, scale=1, size=(100, 2))
    # U=np.random.normal(loc=0,scale=1,size=(100,10))
    x_new = np.tanh(sigmoid(np.dot(W, x.T))).T
    return x_new


if __name__ == '__main__':
    x, y = syn_data_points()
    plot_points(x, y)
    plt.savefig("../data_points_2_dim.png")
    plt.show()
    # plt.close()
    x_high = to_higher_dimension(x)

    pca = PCA(n_components=2)
    nmf=NMF(n_components=2)
    # x_pca = pca.fit_transform(x_high)
    # plot_points(x_pca, y)
    x_nmf=nmf.fit_transform(x_high)
    plot_points(x_nmf,y)
    plt.show()
    # plt.savefig("../data_points_100_dim.png")
