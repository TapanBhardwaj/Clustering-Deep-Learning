import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.engine.topology import Layer, InputSpec
from sklearn import metrics
from sklearn.utils.linear_assignment_ import linear_assignment


def target_distribution(q):
    """
    returns auxilliary distribution from student t distribution
    :param q:
    :return:
    """
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


def cluster_accuracy(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def plot_acc_nmi_ari():
    df = pd.read_csv('../final_weights/idec/har/idec_log.csv', index_col=False)
    df.plot(x='iter', y=['acc', 'nmi', 'ari'])
    plt.title("IDEC on HAR dataset")
    plt.savefig('/Users/tapanbhardwaj/Desktop/IDEC_HAR.png')
    plt.show()


def get_acc_nmi_ari(y, y_pred):
    acc = np.round(cluster_accuracy(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)

    return acc, nmi, ari


class ClusteringLayer(Layer):
    def __init__(self, n_clusters, alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters
