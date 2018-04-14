import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from keras.datasets import mnist
from sklearn.decomposition import PCA

from synthetic_data_generation import syn_data_points


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
    plt.xlabel("h0")
    plt.ylabel("h1")
    plt.legend()
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


def to_higher_dimension(x, y):
    """
    transform data from 2-d to 100-d using non-linear transformation
    assume data points to be 2-dimension
    x=sigmoid(U*sigmoid(W*x(in 2d)))

    :param x:
    :return:
    """
    W = np.random.normal(loc=0, scale=1, size=(10, 2))
    # U=np.random.normal(loc=0,scale=1,size=(100,10))
    x_new = np.square(sigmoid(np.dot(W, x.T))).T
    return x_new, y


def to_higher_dimension2(x, y):
    W = np.random.normal(loc=0, scale=1, size=(10, 2))
    # U=np.random.normal(loc=0,scale=1,size=(100,10))
    x_new = np.tanh(sigmoid(np.dot(W, x.T))).T
    return x_new, y


def load_synthetic():
    x, y = syn_data_points()
    W = np.random.normal(loc=0, scale=1, size=(10, 2))
    U = np.random.normal(loc=0, scale=1, size=(100, 10))

    x_new = sigmoid(np.dot(U, sigmoid(np.dot(W, x.T)))).T

    pickle_file_path = '/Users/tapanbhardwaj/Downloads/github_projects/Clustering-Deep-Learning/Data/synthetic_data.pkl'

    if not os.path.isfile(pickle_file_path):
        with open(pickle_file_path, 'wb') as f:
            pickle.dump([x_new, y], f)

    with open(pickle_file_path, 'rb') as f:
        x, y = pickle.load(f)

    print('SYNTHETIC data samples shape : {}'.format(x_new.shape))
    return x, y


def load_mnist():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    # converting to (784, ) shape
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.0)  # normalize pixel value between 0 and 1
    print('MNIST samples shape : {}'.format(x.shape))
    return x, y


def load_har():
    data = scio.loadmat('../Data/har/HAR.mat')
    x = data['X']
    x = x.astype('float32')
    y = data['Y'] - 1
    x = x[:10200]
    y = y[:10200]
    y = y.reshape((10200,))
    print('HHAR samples shape : {}'.format(x.shape))
    return x, y


if __name__ == '__main__':
    x1, y1 = syn_data_points()
    plot_points(x1, y1)
    plt.title('Plot of 2 dimensional data points')
    plt.savefig("../data_points_2_dim.png")
    plt.close()
    x_high, y = load_synthetic()

    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_high)
    plot_points(x_pca, y)
    plt.title('Plot of data points after PCA')
    plt.savefig("../data_points_100_dim.png")
