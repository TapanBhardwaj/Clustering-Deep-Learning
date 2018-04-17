from time import time
import numpy as np
import pickle
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.cluster import KMeans
from sklearn import metrics
from synthetic_plots import plot_points
import matplotlib.pyplot as plt

np.random.seed(0)


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def autoencoder(dims, act='relu'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        Model of autoencoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks - 1):
        h = Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    # internal layers in decoder
    for i in range(n_stacks - 1, 0, -1):
        h = Dense(dims[i], activation=act, name='decoder_%d' % i)(h)

    # output
    h = Dense(dims[0], name='decoder_0')(h)

    return Model(inputs=x, outputs=h)


class DCN(object):
    def __init__(self,
                 dims,
                 n_clusters=4,
                 alpha=1.0,
                 batch_size=10000):

        super(DCN, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size
        self.autoencoder = autoencoder(self.dims)

    def initialize_model(self, gamma=0.1, optimizer='adam', ae_weights='ae_weights/synthetic_ae_weights.h5'):

        if ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)
            print('Pretrained AE weights are loaded successfully.')
        hidden = self.autoencoder.get_layer(name='encoder_%d' % (self.n_stacks - 1)).output
        self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)

        # prepare DCN model
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[self.encoder.output, self.autoencoder.output])
        self.model.compile(loss={'encoder_2': 'mse', 'decoder_0': 'mse'},
                           loss_weights=[gamma, 1],
                           optimizer=optimizer)

    def load_weights(self, weights_path):  # load weights of DCN model
        self.model.load_weights(weights_path)

    def clustering(self, x, y=None,
                   maxiter=100):
        loss = [0, 0, 0]
        for ite in range(int(maxiter)):
            q, _ = self.model.predict(x, verbose=0)  # actual representation in k-means friendly space
            p, _ = self.model.predict(x, verbose=0)
            km = KMeans(n_clusters=self.n_clusters).fit(q)
            centers = km.cluster_centers_
            labels = km.labels_
            for i in range(q.shape[0]):
                q[i, :] = centers[labels[i], :]

            if y is not None:
                acc = np.round(cluster_acc(y, labels), 5)
                nmi = np.round(metrics.normalized_mutual_info_score(y, labels), 5)
                ari = np.round(metrics.adjusted_rand_score(y, labels), 5)
                loss = np.round(loss, 5)
                print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)
                if (acc > 0.95):
                    break
            loss = self.model.train_on_batch(x=x, y=[q, x])

        return labels, p


if __name__ == "__main__":
    optimizer = SGD(lr=0.1, momentum=0.99)

    with open('synthetic_data.pkl', 'rb') as f:
        x, y = pickle.load(f)
    print('Data loaded successfully')
    print(x.shape)
    # prepare the DCN model
    dcn = DCN(dims=[x.shape[-1], 50, 10, 2], n_clusters=4)
    dcn.initialize_model(ae_weights='ae_weights/synthetic_ae_weights.h5', optimizer=optimizer)

    dcn.model.summary()

    # begin clustering, time not include pretraining part.
    t0 = time()
    y_pred, x_pred = dcn.clustering(x=x, y=y, maxiter=100)
    plot_points(x_pred, y)
    print('acc:', cluster_acc(y, y_pred))
    print('nmi', metrics.normalized_mutual_info_score(y, y_pred))
    print('rand', metrics.adjusted_rand_score(y, y_pred))
    print('clustering time: ', (time() - t0))
